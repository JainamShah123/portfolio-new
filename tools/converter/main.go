package main

import (
	"bufio"
	"bytes"
	"fmt"
	"html/template"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/extension"
	"github.com/yuin/goldmark/parser"
	"github.com/yuin/goldmark/renderer/html"
)

// Global pool to reuse buffers across goroutines
var bufPool = sync.Pool{
	New: func() any {
		return new(bytes.Buffer)
	},
}

type Post struct {
	ID          string
	Title       string
	Date        string
	ParsedDate  time.Time
	Tags        []string
	Description string
	Content     template.HTML
	OutputPath  string
	RelativeURL string
}

func main() {
	start := time.Now()

	// 1. Configuration & Setup
	inputDir := filepath.Clean("../../blogs")
	outputDir := filepath.Clean("../../dist")
	templatePath := filepath.Clean("template.html")
	indexPaths := []string{filepath.Clean("../../index.html")}

	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}

	tmpl := template.Must(template.ParseFiles(templatePath))
	md := goldmark.New(
		goldmark.WithExtensions(extension.Table),
		goldmark.WithParserOptions(parser.WithAutoHeadingID()),
		goldmark.WithRendererOptions(html.WithUnsafe()),
	)

	// 2. Discover files first to pre-allocate memory
	var files []string
	filepath.WalkDir(inputDir, func(path string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && strings.HasSuffix(path, ".md") {
			files = append(files, path)
		}
		return nil
	})

	numFiles := len(files)
	posts := make([]Post, 0, numFiles) // Pre-allocated capacity
	var postsMu sync.Mutex

	// 3. Concurrency Setup
	jobs := make(chan string, numFiles)
	var wg sync.WaitGroup
	numWorkers := runtime.NumCPU()

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for path := range jobs {
				post, err := processFile(path, tmpl, md, outputDir)
				if err != nil {
					log.Printf("Error processing %s: %v", path, err)
					continue
				}
				postsMu.Lock()
				posts = append(posts, post)
				postsMu.Unlock()
			}
		}()
	}

	// Send jobs
	for _, path := range files {
		jobs <- path
	}
	close(jobs)
	wg.Wait()

	// 4. Sort and Finalize
	sort.Slice(posts, func(i, j int) bool {
		return posts[i].ParsedDate.After(posts[j].ParsedDate)
	})

	for _, indexPath := range indexPaths {
		if err := updateIndex(indexPath, posts); err != nil {
			log.Printf("Error updating index %s: %v", indexPath, err)
		}
	}

	fmt.Printf("Processed %d posts in %v\n", len(posts), time.Since(start))
}

func processFile(path string, tmpl *template.Template, md goldmark.Markdown, outputDir string) (Post, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Post{}, err
	}

	meta, body := parseMarkdown(raw)

	// Use pool for MD -> HTML conversion
	buf := bufPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer bufPool.Put(buf)

	if err := md.Convert(body, buf); err != nil {
		return Post{}, err
	}

	htmlContent := fastProcessQuoteCards(buf.String())

	// Date parsing
	dateStr := meta["date"]
	parsedDate, _ := time.Parse("January 2, 2006", dateStr)

	var tags []string
	if t, ok := meta["tags"]; ok {
		for _, tag := range strings.Split(t, ",") {
			tags = append(tags, strings.TrimSpace(tag))
		}
	}

	base := filepath.Base(path)
	name := strings.TrimSuffix(base, filepath.Ext(base))
	outPath := filepath.Join(outputDir, name+".html")

	id := meta["id"]
	if id == "" {
		id = name // fallback to filename
	}

	post := Post{
		ID:          id,
		Title:       meta["title"],
		Date:        dateStr,
		ParsedDate:  parsedDate,
		Tags:        tags,
		Description: meta["description"],
		Content:     template.HTML(htmlContent),
		OutputPath:  outPath,
		RelativeURL: "dist/" + name + ".html",
	}

	// Optimized File Write
	f, err := os.Create(outPath)
	if err != nil {
		return post, err
	}
	defer f.Close()

	// Use a buffered writer to avoid constant syscalls during template execution
	writer := bufio.NewWriter(f)
	if err := tmpl.Execute(writer, post); err != nil {
		return post, err
	}
	writer.Flush()

	return post, nil
}

// fastProcessQuoteCards avoids line-by-line scanning where possible
func fastProcessQuoteCards(input string) string {
	// Global replacements
	r := strings.NewReplacer(
		"<blockquote>", `<blockquote class="quote-card">`,
		"<p>—", "<footer>—",
		"<p>--", "<footer>--",
	)
	output := r.Replace(input)

	// If we replaced a footer start, we must replace the corresponding end
	if strings.Contains(output, "<footer>") {
		output = strings.ReplaceAll(output, "—</p>", "—</footer>")
		output = strings.ReplaceAll(output, "--</p>", "--</footer>")
	}
	return output
}

func parseMarkdown(content []byte) (map[string]string, []byte) {
	meta := make(map[string]string)
	if !bytes.HasPrefix(content, []byte("---\n")) && !bytes.HasPrefix(content, []byte("---\r\n")) {
		return meta, content
	}

	endIdx := bytes.Index(content[3:], []byte("\n---"))
	if endIdx == -1 {
		return meta, content
	}
	endIdx += 3

	frontmatter := content[3:endIdx]
	body := bytes.TrimLeft(content[endIdx+4:], "\r\n")

	scanner := bufio.NewScanner(bytes.NewReader(frontmatter))
	for scanner.Scan() {
		line := scanner.Text()
		if key, val, found := strings.Cut(line, ":"); found {
			meta[strings.TrimSpace(key)] = strings.TrimSpace(val)
		}
	}
	return meta, body
}

func updateIndex(path string, posts []Post) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	startMarker := []byte("<!-- BLOGS_START -->")
	endMarker := []byte("<!-- BLOGS_END -->")

	startIdx := bytes.Index(content, startMarker)
	endIdx := bytes.Index(content, endMarker)

	if startIdx == -1 || endIdx == -1 {
		return fmt.Errorf("markers missing in %s", path)
	}

	// Use pool for building the index HTML
	resBuf := bufPool.Get().(*bytes.Buffer)
	resBuf.Reset()
	defer bufPool.Put(resBuf)

	resBuf.Write(content[:startIdx+len(startMarker)])
	resBuf.WriteString("\n")

	for _, p := range posts {
		fmt.Fprintf(resBuf, `
            <div class="log-item" id="%s">
                <a href="%s">
                    <div class="log-header">
                        <h3>%s</h3>
                        <span class="log-date">%s</span>
                    </div>
                    <p>%s</p>
                    <span class="log-meta">%s</span>
                </a>
            </div>`, p.ID, p.RelativeURL, p.Title, p.Date, p.Description, strings.Join(p.Tags, " · "))
	}

	resBuf.WriteString("\n            ")
	resBuf.Write(content[endIdx:])

	return os.WriteFile(path, resBuf.Bytes(), 0644)
}

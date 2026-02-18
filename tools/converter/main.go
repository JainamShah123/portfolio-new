package main

import (
	"bufio"
	"bytes"
	"fmt"
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

// ─── 1. Core Logic & Data Structures ──────────────────────────────────────────

type Post struct {
	ID          string
	Title       string
	Date        string
	ParsedDate  time.Time
	Tags        []string
	Description string
	Content     []byte
	RelativeURL string
	IndexHTML   string // Pre-rendered HTML for the log-item
}

// Global buffer pool to reduce GC pressure (reused for Markdown rendering)
var bufPool = sync.Pool{
	New: func() any { return bytes.NewBuffer(make([]byte, 0, 64*1024)) },
}

// Global writer pool (reused for File I/O)
var writerPool = sync.Pool{
	New: func() any { return bufio.NewWriterSize(nil, 64*1024) },
}

// ─── 2. The Zero-Reflection Template Engine ───────────────────────────────────

type CompiledTemplate struct {
	// Stores the static HTML parts between the dynamic variables
	seg [6][]byte
}

var placeholders = []string{
	"{{.Title}}",
	"{{.Title}}", // Assumes Title appears twice (e.g., <title> and <h1>)
	`{{range $i, $tag := .Tags}}{{if $i}} · {{end}}{{$tag}}{{end}}`,
	"{{.Date}}",
	"{{.Content}}",
}

func compileTemplate(path string) (*CompiledTemplate, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	ct := &CompiledTemplate{}
	remaining := raw

	for i, ph := range placeholders {
		idx := bytes.Index(remaining, []byte(ph))
		if idx == -1 {
			return nil, fmt.Errorf("template mismatch: missing placeholder %q", ph)
		}
		// Copy data to avoid referencing the original 'raw' array
		ct.seg[i] = append([]byte(nil), remaining[:idx]...)
		remaining = remaining[idx+len(ph):]
	}
	ct.seg[5] = append([]byte(nil), remaining...)
	return ct, nil
}

func (ct *CompiledTemplate) render(post Post, w *bufio.Writer) {
	// Sequential high-speed writes
	w.Write(ct.seg[0])
	w.WriteString(post.Title)
	w.Write(ct.seg[1])
	w.WriteString(post.Title)
	w.Write(ct.seg[2])

	// Optimized Tag Join
	for i, tag := range post.Tags {
		if i > 0 {
			w.WriteString(" · ")
		}
		w.WriteString(tag)
	}

	w.Write(ct.seg[3])
	w.WriteString(post.Date)
	w.Write(ct.seg[4])
	w.Write(post.Content)
	w.Write(ct.seg[5])
}

// ─── 3. Main Execution Flow ───────────────────────────────────────────────────

func main() {
	start := time.Now()

	// Config
	inputDir := filepath.Clean("../../blogs")
	outputDir := filepath.Clean("../../dist")
	templatePath := filepath.Clean("template.html")
	indexPaths := []string{filepath.Clean("../../index.html")}

	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Fatalf("Failed to create dist: %v", err)
	}

	tmpl, err := compileTemplate(templatePath)
	if err != nil {
		log.Fatalf("Template Error: %v", err)
	}

	// Goldmark is thread-safe, init once
	md := goldmark.New(
		goldmark.WithExtensions(extension.Table),
		goldmark.WithParserOptions(parser.WithAutoHeadingID()),
		goldmark.WithRendererOptions(html.WithUnsafe()),
	)

	// Discover files
	var files []string
	filepath.WalkDir(inputDir, func(path string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && strings.HasSuffix(path, ".md") {
			files = append(files, path)
		}
		return nil
	})

	numFiles := len(files)
	posts := make([]Post, 0, numFiles)
	var postsMu sync.Mutex

	// Worker Pool
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
					log.Printf("Skipping %s: %v", path, err)
					continue
				}
				postsMu.Lock()
				posts = append(posts, post)
				postsMu.Unlock()
			}
		}()
	}

	for _, path := range files {
		jobs <- path
	}
	close(jobs)
	wg.Wait()

	// Sort & Index
	sort.Slice(posts, func(i, j int) bool {
		return posts[i].ParsedDate.After(posts[j].ParsedDate)
	})

	for _, indexPath := range indexPaths {
		if err := updateIndex(indexPath, posts); err != nil {
			log.Printf("Index Update Error: %v", err)
		}
	}

	fmt.Printf("Processed %d posts in %v\n", len(posts), time.Since(start))
}

// ─── 4. File Processing (The Hot Path) ────────────────────────────────────────

func processFile(path string, tmpl *CompiledTemplate, md goldmark.Markdown, outputDir string) (Post, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Post{}, err
	}

	meta, body := parseMarkdown(raw)

	// MD -> HTML
	buf := bufPool.Get().(*bytes.Buffer)
	buf.Reset()

	// We handle defer manually to ensure buffer is returned even if we error early
	// but here we just defer cleanly for safety.
	defer bufPool.Put(buf)

	if err := md.Convert(body, buf); err != nil {
		return Post{}, err
	}

	// Byte-level post-processing
	htmlBytes := fastProcessQuoteCards(buf.Bytes())

	// Meta parsing
	base := filepath.Base(path)
	name := strings.TrimSuffix(base, filepath.Ext(base))

	// Date Logic
	pDate, _ := time.Parse("2006-01-02", meta["date"])
	if pDate.IsZero() {
		pDate, _ = time.Parse("January 2, 2006", meta["date"])
	}

	// Tag Logic (Zero-alloc split could go here, but strings.Split is fast enough for short tags)
	var tags []string
	if tVal := meta["tags"]; tVal != "" {
		parts := strings.Split(tVal, ",")
		tags = make([]string, 0, len(parts))
		for _, t := range parts {
			if trimmed := strings.TrimSpace(t); trimmed != "" {
				tags = append(tags, trimmed)
			}
		}
	}

	post := Post{
		ID:          meta["id"],
		Title:       meta["title"],
		Date:        meta["date"],
		ParsedDate:  pDate,
		Tags:        tags,
		Description: meta["description"],
		Content:     htmlBytes, // No copy, points to new slice from fastProcess
		RelativeURL: "dist/" + name + ".html",
	}
	if post.ID == "" {
		post.ID = name
	}

	// Render the index snippet while the CPU is already hot
	var tagBuilder strings.Builder
	for i, t := range post.Tags {
		if i > 0 {
			tagBuilder.WriteString(" · ")
		}
		tagBuilder.WriteString(t)
	}

	post.IndexHTML = fmt.Sprintf(`
            <div class="log-item" id="%s">
                <a href="%s">
                    <div class="log-header">
                        <h3>%s</h3>
                        <span class="log-date">%s</span>
                    </div>
                    <p>%s</p>
                    <span class="log-meta">%s</span>
                </a>
            </div>`, post.ID, post.RelativeURL, post.Title, post.Date, post.Description, tagBuilder.String())

	// File Write
	f, err := os.OpenFile(filepath.Join(outputDir, name+".html"), os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return post, err
	}

	w := writerPool.Get().(*bufio.Writer)
	w.Reset(f)
	tmpl.render(post, w)
	w.Flush()
	f.Close() // Close file explicitly before putting writer back
	writerPool.Put(w)

	return post, nil
}

// ─── 5. Helpers (Micro-Optimized) ─────────────────────────────────────────────

func parseMarkdown(content []byte) (map[string]string, []byte) {
	meta := make(map[string]string)

	// Fast check for Frontmatter
	if !bytes.HasPrefix(content, []byte("---\n")) && !bytes.HasPrefix(content, []byte("---\r\n")) {
		return meta, content
	}

	// Find end of frontmatter
	endIdx := bytes.Index(content[3:], []byte("\n---"))
	if endIdx == -1 {
		return meta, content
	}
	endIdx += 3 // Adjust for the offset

	// Zero-alloc parsing using bytes.Cut (Go 1.18+)
	// We avoid bufio.Scanner and strictly process the slice
	rem := content[3:endIdx]
	for len(rem) > 0 {
		var line []byte
		// Find next newline
		if idx := bytes.IndexByte(rem, '\n'); idx >= 0 {
			line = rem[:idx]
			rem = rem[idx+1:]
		} else {
			line = rem
			rem = nil
		}

		// Handle Windows CR
		line = bytes.TrimSuffix(line, []byte("\r"))

		if key, val, found := bytes.Cut(line, []byte(":")); found {
			k := string(bytes.TrimSpace(key))
			v := string(bytes.TrimSpace(val))
			meta[k] = v
		}
	}

	return meta, bytes.TrimLeft(content[endIdx+4:], "\r\n")
}

func fastProcessQuoteCards(input []byte) []byte {
	// We chain replacements. Note: bytes.ReplaceAll allocates a new slice.
	// For 1000 files, this is acceptable.
	// To go faster requires a custom single-pass byte processor,
	// but that is complex to maintain.
	out := bytes.ReplaceAll(input, []byte("<blockquote>"), []byte(`<blockquote class="quote-card">`))

	// Only look for footer replacements if we see a paragraph start (optimization)
	if bytes.Contains(out, []byte("<p>")) {
		out = bytes.ReplaceAll(out, []byte("<p>\u2014"), []byte("<footer>\u2014"))
		out = bytes.ReplaceAll(out, []byte("<p>--"), []byte("<footer>--"))

		if bytes.Contains(out, []byte("<footer>")) {
			out = bytes.ReplaceAll(out, []byte("\u2014</p>"), []byte("\u2014</footer>"))
			out = bytes.ReplaceAll(out, []byte("--</p>"), []byte("--</footer>"))
		}
	}
	return out
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

	// Reusing the global buffer pool for the index generation
	buf := bufPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer bufPool.Put(buf)

	buf.Write(content[:startIdx+len(startMarker)])
	buf.WriteByte('\n')

	// Efficient String Builder pattern
	for _, p := range posts {
		buf.WriteString(p.IndexHTML)
	}

	buf.WriteString("\n            ")
	buf.Write(content[endIdx:])

	return os.WriteFile(path, buf.Bytes(), 0644)
}

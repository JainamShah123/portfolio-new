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
	Styles      string // Inlined CSS
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
	seg [7][]byte
}

var placeholders = []string{
	"{{.Title}}",
	"{{.Styles}}",
	"{{.Title}}",
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
	ct.seg[6] = append([]byte(nil), remaining...)
	return ct, nil
}

func (ct *CompiledTemplate) render(post Post, w *bufio.Writer) {
	// Sequential high-speed writes
	w.Write(ct.seg[0])
	w.WriteString(post.Title)
	w.Write(ct.seg[1])
	w.WriteString(post.Styles)
	w.Write(ct.seg[2])
	w.WriteString(post.Title)
	w.Write(ct.seg[3])

	// Optimized Tag Join
	for i, tag := range post.Tags {
		if i > 0 {
			w.WriteString(" · ")
		}
		w.WriteString(tag)
	}

	w.Write(ct.seg[4])
	w.WriteString(post.Date)
	w.Write(ct.seg[5])
	w.Write(post.Content)
	w.Write(ct.seg[6])
}

// ─── 3. Main Execution Flow ───────────────────────────────────────────────────

func main() {
	start := time.Now()

	// Config
	inputDir := filepath.Clean("../../blogs")
	outputDir := filepath.Clean("../../")
	templatePath := filepath.Clean("template.html")
	stylesPath := filepath.Clean("../../styles.css")
	indexPaths := []string{filepath.Clean("../../index.html")}

	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Fatalf("Failed to create dist: %v", err)
	}

	stylesRaw, err := os.ReadFile(stylesPath)
	if err != nil {
		log.Fatalf("Styles Error: %v", err)
	}
	styles := stripCSSComments(string(stylesRaw))

	tmplRaw, err := os.ReadFile(templatePath)
	if err != nil {
		log.Fatalf("Template Error: %v", err)
	}
	templateContent := string(tmplRaw)

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
				post, err := processFile(path, tmpl, md, outputDir, styles, templateContent)
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
		if err := updateIndex(indexPath, posts, styles); err != nil {
			log.Printf("Index Update Error: %v", err)
		}
	}

	fmt.Printf("Processed %d posts in %v\n", len(posts), time.Since(start))
}

// ─── 4. File Processing (The Hot Path) ────────────────────────────────────────

func processFile(path string, tmpl *CompiledTemplate, md goldmark.Markdown, outputDir string, fullCSS string, templateHTML string) (Post, error) {
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
	htmlBytes = fastProcessAssetPaths(htmlBytes)

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
		RelativeURL: name + ".html",
		Styles:      optimizeCSS(fullCSS, templateHTML+string(htmlBytes)),
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
	out := bytes.ReplaceAll(input, []byte("<blockquote>"), []byte(`<blockquote class="quote-card">`))

	if bytes.Contains(out, []byte("<p>")) {
		// Replace start of footer
		out = bytes.ReplaceAll(out, []byte("<p>\u2014"), []byte("<footer>\u2014"))
		out = bytes.ReplaceAll(out, []byte("<p>--"), []byte("<footer>--"))

		// Robustly replace the corresponding closing tag
		// We look for <footer> and find the next </p>
		if bytes.Contains(out, []byte("<footer>")) {
			var result bytes.Buffer
			remaining := out
			for {
				footerIdx := bytes.Index(remaining, []byte("<footer>"))
				if footerIdx == -1 {
					result.Write(remaining)
					break
				}
				// Write up to the end of <footer>...
				closeTagStart := bytes.Index(remaining[footerIdx:], []byte("</p>"))
				if closeTagStart != -1 {
					closeTagStart += footerIdx
					result.Write(remaining[:closeTagStart])
					result.WriteString("</footer>")
					remaining = remaining[closeTagStart+4:]
				} else {
					result.Write(remaining)
					break
				}
			}
			out = result.Bytes()
		}
	}
	return out
}

func fastProcessAssetPaths(input []byte) []byte {
	return bytes.ReplaceAll(input, []byte("../assets/"), []byte("assets/"))
}

func updateIndex(path string, posts []Post, fullCSS string) error {
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

	// Inline CSS for Index
	finalHTML := buf.Bytes()
	cssStart := bytes.Index(finalHTML, []byte("<!-- CSS_START -->"))
	cssEnd := bytes.Index(finalHTML, []byte("<!-- CSS_END -->"))

	if cssStart != -1 && cssEnd != -1 {
		optimized := optimizeCSS(fullCSS, string(finalHTML))
		var result bytes.Buffer
		result.Write(finalHTML[:cssStart])
		result.WriteString("<style>\n")
		result.WriteString(optimized)
		result.WriteString("\n    </style>")
		result.Write(finalHTML[cssEnd+len("<!-- CSS_END -->"):])
		finalHTML = result.Bytes()
	}

	return os.WriteFile(path, finalHTML, 0644)
}

// ─── 6. CSS Optimization ─────────────────────────────────────────────────────

func optimizeCSS(css string, html string) string {
	// Simple CSS Purger
	// 1. Minify (sort of)
	css = strings.ReplaceAll(css, "\r", "")

	var sb strings.Builder
	remaining := css

	for {
		idx := strings.Index(remaining, "{")
		if idx == -1 {
			break
		}
		selectors := strings.TrimSpace(remaining[:idx])
		endIdx := strings.Index(remaining[idx:], "}")
		if endIdx == -1 {
			break
		}
		endIdx += idx
		body := remaining[idx : endIdx+1]
		remaining = remaining[endIdx+1:]

		// Handle @media - we just include them if they contain used selectors
		if strings.HasPrefix(selectors, "@media") {
			// Extract rules inside @media
			inner := body[1 : len(body)-1]
			optimizedInner := optimizeCSS(inner, html)
			if optimizedInner != "" {
				sb.WriteString(selectors)
				sb.WriteString(" {\n")
				sb.WriteString(optimizedInner)
				sb.WriteString("\n}\n")
			}
			continue
		}

		// Check if any of the selectors is used
		used := false
		parts := strings.Split(selectors, ",")
		for _, sel := range parts {
			if isSelectorUsed(strings.TrimSpace(sel), html) {
				used = true
				break
			}
		}

		if used {
			sb.WriteString(selectors)
			sb.WriteString(" ")
			sb.WriteString(body)
			sb.WriteString("\n")
		}
	}

	return strings.TrimSpace(sb.String())
}

func isSelectorUsed(selector string, html string) bool {
	// Very naive but effective for this specific project
	// Check for tags and classes
	if selector == "*" || selector == ":root" || selector == "body" || selector == "main" || selector == "html" || selector == "header" || selector == "footer" {
		return true
	}

	// Remove pseudo-classes and pseudo-elements
	if idx := strings.Index(selector, ":"); idx != -1 {
		selector = selector[:idx]
	}

	// Remove attribute selectors
	if idx := strings.Index(selector, "["); idx != -1 {
		selector = selector[:idx]
	}

	selector = strings.TrimSpace(selector)
	if selector == "" {
		return true
	}

	// Split by space for descendant selectors
	parts := strings.Fields(selector)
	for _, p := range parts {
		// handle combined selectors like h1.title
		if strings.Contains(p, ".") {
			sub := strings.Split(p, ".")
			// sub[0] is tag (optional), sub[1:] are classes
			if sub[0] != "" && !strings.Contains(html, "<"+sub[0]) {
				return false
			}
			for i := 1; i < len(sub); i++ {
				if !hasClass(html, sub[i]) {
					return false
				}
			}
			continue
		}

		// If it's a class
		if strings.HasPrefix(p, ".") {
			if !hasClass(html, p[1:]) {
				return false
			}
			continue
		}
		// If it's an ID
		if strings.HasPrefix(p, "#") {
			id := p[1:]
			if !strings.Contains(html, `id="`+id+`"`) {
				return false
			}
			continue
		}
		// If it's a tag
		if !strings.Contains(strings.ToLower(html), "<"+strings.ToLower(p)) {
			return false
		}
	}

	return true
}

func hasClass(html string, className string) bool {
	return strings.Contains(html, `class="`+className+`"`) ||
		strings.Contains(html, ` "`+className+`"`) ||
		strings.Contains(html, `class="`+className+` `) ||
		strings.Contains(html, ` "`+className+` `)
}

func stripCSSComments(css string) string {
	for {
		start := strings.Index(css, "/*")
		if start == -1 {
			break
		}
		end := strings.Index(css[start:], "*/")
		if end == -1 {
			break
		}
		css = css[:start] + css[start+end+2:]
	}
	return css
}


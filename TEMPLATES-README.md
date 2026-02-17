# HTML Template Library for Dev Logs

## 📚 Quick Reference

Copy-paste HTML blocks for your dev logs. All components are modular, nestable, and styled automatically.

---

## 🎯 Essential Templates

### Log Structure
```html
<article>
    <header>
        <h1>Log Title</h1>
        <div class="article-meta">
            <span class="log-meta">Tags · Here</span>
            <span class="log-date">Month Year</span>
        </div>
    </header>

    <!-- Your content sections -->
</article>
```

---

## 📝 Text

```html
<!-- Section -->
<h2>Section Title</h2>
<p>Paragraph text.</p>

<!-- Emphasis -->
<strong>Bold</strong>
<em>Italic</em>
<mark>Highlight</mark>
<code>inline code</code>

<!-- Basic Quote -->
<blockquote>
    <p>Quoted text</p>
</blockquote>

<!-- Quote Card with Author -->
<blockquote class="quote-card">
    <p>The quote text goes here.</p>
    <footer>— Author Name</footer>
</blockquote>
```

---

## 📋 Lists

```html
<!-- Bullets -->
<ul>
    <li>Item one</li>
    <li>Item two</li>
</ul>

<!-- Numbered -->
<ol>
    <li>Step one</li>
    <li>Step two</li>
</ol>

<!-- Nested -->
<ul>
    <li>Parent
        <ul>
            <li>Child</li>
        </ul>
    </li>
</ul>
```

---

## 💻 Code

```html
<!-- Inline -->
Use <code>methodName()</code> for inline code.

<!-- Block -->
<pre><code>function example() {
    return "code here";
}
</code></pre>

<!-- With Header -->
<div class="code-block">
    <div class="code-header">
        <span class="code-file">filename.js</span>
        <span class="code-lang">JavaScript</span>
    </div>
    <pre><code>const x = 42;
</code></pre>
</div>
```

---

## 🖼️ Media

```html
<!-- Image -->
<img src="images/screenshot.png" alt="Description">

<!-- With Caption -->
<figure>
    <img src="images/screenshot.png" alt="Description">
    <figcaption>Caption text</figcaption>
</figure>

<!-- Video -->
<figure>
    <video controls>
        <source src="videos/demo.mp4" type="video/mp4">
    </video>
    <figcaption>Video description</figcaption>
</figure>

<!-- Side-by-side -->
<div class="media-grid">
    <figure>
        <img src="images/before.png" alt="Before">
        <figcaption>Before</figcaption>
    </figure>
    <figure>
        <img src="images/after.png" alt="After">
        <figcaption>After</figcaption>
    </figure>
</div>
```

---

## 💡 Callouts

```html
<!-- Default -->
<div class="callout">
    <strong>Label:</strong>
    <p>Important point here.</p>
</div>

<!-- Warning -->
<div class="callout callout-warning">
    <strong>Warning:</strong>
    <p>Careful with this approach.</p>
</div>

<!-- Success -->
<div class="callout callout-success">
    <strong>Result:</strong>
    <p>Performance improved 60%.</p>
</div>

<!-- Info -->
<div class="callout callout-info">
    <strong>Note:</strong>
    <p>Requires version 3.0+</p>
</div>
```

---

## 📊 Tables

```html
<table>
    <thead>
        <tr>
            <th>Column 1</th>
            <th>Column 2</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Data 1</td>
            <td>Data 2</td>
        </tr>
    </tbody>
</table>
```

---

## 🔗 Links

```html
<!-- Inline -->
<a href="https://example.com">Link text</a>

<!-- External -->
<a href="https://example.com" target="_blank" rel="noopener">Opens in new tab</a>

<!-- Link list -->
<ul class="link-list">
    <li><a href="#">Resource 1</a></li>
    <li><a href="#">Resource 2</a></li>
</ul>
```

---

## 🏗️ Layout

```html
<!-- Divider -->
<hr>

<!-- Two columns -->
<div class="two-columns">
    <div>
        <h3>Left</h3>
        <p>Content</p>
    </div>
    <div>
        <h3>Right</h3>
        <p>Content</p>
    </div>
</div>

<!-- Collapsible -->
<details>
    <summary>Click to expand</summary>
    <p>Hidden content</p>
</details>
```

---

## 🎨 Common Patterns

### Problem-Solution Pattern
```html
<h2>The Problem</h2>
<p>Description of the issue...</p>

<div class="callout callout-warning">
    <strong>Issue:</strong>
    <p>Specific problem details.</p>
</div>

<h2>Solution</h2>
<p>How it was solved...</p>

<div class="code-block">
    <div class="code-header">
        <span class="code-file">solution.dart</span>
        <span class="code-lang">Dart</span>
    </div>
    <pre><code>// Solution code
</code></pre>
</div>

<h2>Results</h2>
<div class="callout callout-success">
    <strong>Impact:</strong>
    <p>Performance improved by 60%</p>
</div>
```

### Before/After Comparison
```html
<h2>Before vs After</h2>

<div class="media-grid">
    <figure>
        <video controls>
            <source src="videos/before.mp4" type="video/mp4">
        </video>
        <figcaption>Before: 15fps, visible jank</figcaption>
    </figure>
    <figure>
        <video controls>
            <source src="videos/after.mp4" type="video/mp4">
        </video>
        <figcaption>After: Smooth 60fps</figcaption>
    </figure>
</div>

<table>
    <thead>
        <tr>
            <th>Metric</th>
            <th>Before</th>
            <th>After</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>FPS</td>
            <td>15</td>
            <td>60</td>
        </tr>
        <tr>
            <td>Memory</td>
            <td>2MB/s</td>
            <td>200KB/s</td>
        </tr>
    </tbody>
</table>
```

### Code Explanation Pattern
```html
<p>The key is using <code>ValueListenableBuilder</code>:</p>

<div class="code-block">
    <div class="code-header">
        <span class="code-file">widget.dart</span>
        <span class="code-lang">Dart</span>
    </div>
    <pre><code>ValueListenableBuilder<Price>(
  valueListenable: priceNotifier,
  builder: (_, price, __) => Text('$${price.value}'),
)
</code></pre>
</div>

<p>This ensures <em>only this widget</em> rebuilds.</p>

<div class="callout">
    <strong>Why this works:</strong>
    <p>Each widget listens to its own notifier, preventing cascade rebuilds.</p>
</div>
```

---

## 📄 Complete Starter Template

Open [`templates.html`](templates.html) for the full comprehensive reference with all components and examples.

---

## 💡 Tips

**Keep it simple**: Use semantic HTML, let the CSS handle styling  
**Nest freely**: All components work inside each other  
**Stay consistent**: Stick to the same patterns across logs  
**Focus on content**: These templates let you focus on writing, not layout

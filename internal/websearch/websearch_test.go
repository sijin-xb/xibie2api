package websearch

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"
)

func TestSearchSearxng(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/search" || r.URL.Query().Get("format") != "json" {
			t.Errorf("unexpected request: %s", r.URL)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"results":[
			{"url":"https://a.example/1","title":"A","content":"aa","publishedDate":"2026-01-01T00:00:00Z"},
			{"url":"https://b.example/2","title":"B","content":"bb"}
		]}`))
	}))
	defer srv.Close()

	c := NewClient(Config{Enabled: true, Backend: "searxng", BaseURL: srv.URL, MaxResults: 5})
	res, err := c.Search(context.Background(), "hello", 5, nil, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(res) != 2 || res[0].URL != "https://a.example/1" || res[0].PageAge == "" {
		t.Fatalf("res=%+v", res)
	}
}

func TestSearchSearxngRequiresBaseURL(t *testing.T) {
	c := NewClient(Config{Enabled: true, Backend: "searxng"})
	if _, err := c.Search(context.Background(), "q", 3, nil, nil); err == nil {
		t.Fatal("expected error for missing base_url")
	}
}

func TestSearchBraveHTML(t *testing.T) {
	html := `<main><div class="snippet svelte-jmfu5f" data-pos="0" data-type="web" data-keynav="true">
		<div class="result-body"><div class="result-wrapper"><div class="result-content">
		<a href="https://go.dev/doc/go1.24" target="_self" class="svelte-14r20fy l1">
		<div class="site-name-wrapper"><div class="url-wrapper"><cite class="snippet-url">go.dev</cite></div></div>
		<div class="title search-snippet-title line-clamp-1">Go 1.24 Release Notes</div>
		</a><div class="generic-snippet"><div class="content desktop-default-regular t-primary line-clamp-dynamic">Go 1.24 is a major release &amp; adds features.</div></div>
		</div></div></div></div>
		<div class="snippet svelte-jmfu5f" data-pos="1" data-type="web">
		<div class="result-content"><a href="https://example.com/b">Second Result</a>
		<div class="title search-snippet-title">Second &lt;Title&gt;</div>
		<div class="content desktop-default-regular t-primary line-clamp-dynamic">Second desc</div>
		</div></div></main>`
	res := parseBraveHTML(html, 5)
	if len(res) != 2 {
		t.Fatalf("res=%+v", res)
	}
	if res[0].URL != "https://go.dev/doc/go1.24" || res[0].Title != "Go 1.24 Release Notes" || res[0].Snippet != "Go 1.24 is a major release & adds features." {
		t.Fatalf("res[0]=%+v", res[0])
	}
	if res[1].URL != "https://example.com/b" || res[1].Title != "Second <Title>" {
		t.Fatalf("res[1]=%+v", res[1])
	}
}

func TestSearchBingRSS(t *testing.T) {
	rss := `<?xml version="1.0" encoding="utf-8"?><rss version="2.0"><channel>
		<title>Bing: x</title>
		<item><title>标题 A</title><link>https://a.example/x</link><description>摘要 A &amp; more</description></item>
		<item><title>Title B</title><link>https://b.example/y</link><description>Snippet B</description></item>
	</channel></rss>`
	res := parseBingRSS(rss, 5)
	if len(res) != 2 {
		t.Fatalf("res=%+v", res)
	}
	if res[0].URL != "https://a.example/x" || res[0].Title != "标题 A" || res[0].Snippet != "摘要 A & more" {
		t.Fatalf("res[0]=%+v", res[0])
	}
	if res[1].Title != "Title B" {
		t.Fatalf("res[1]=%+v", res[1])
	}
}

func TestSearchDuckDuckGoHTML(t *testing.T) {
	html := `<div>
		<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fc.example%2Fz&rut=1">标题 C</a>
		<a class="result__snippet" href="x">摘要 C</a>
		<a rel="nofollow" class="result__a" href="https://d.example/direct">Direct D</a>
		<a class="result__snippet">Snippet D</a>
	</div>`
	res := parseDDGHTML(html, 5)
	if len(res) != 2 {
		t.Fatalf("res=%+v", res)
	}
	if res[0].URL != "https://c.example/z" || res[0].Snippet != "摘要 C" {
		t.Fatalf("res[0]=%+v", res[0])
	}
	if res[1].URL != "https://d.example/direct" {
		t.Fatalf("res[1]=%+v", res[1])
	}
}

func TestFilterDomains(t *testing.T) {
	in := []Result{
		{URL: "https://a.example/1"},
		{URL: "https://sub.b.example/2"},
		{URL: "https://c.example/3"},
	}
	got := filterDomains(in, nil, []string{"b.example"})
	if len(got) != 2 || got[0].URL != "https://a.example/1" {
		t.Fatalf("blocked filter got=%+v", got)
	}
	got = filterDomains(in, []string{"b.example"}, nil)
	if len(got) != 1 || got[0].URL != "https://sub.b.example/2" {
		t.Fatalf("allowed filter got=%+v", got)
	}
}

func TestDisabledClient(t *testing.T) {
	c := NewClient(Config{Enabled: false, Backend: "bing"})
	if c.Enabled() {
		t.Fatal("expected disabled")
	}
	if _, err := c.Search(context.Background(), "q", 3, nil, nil); err == nil {
		t.Fatal("expected error when disabled")
	}
}

// TestSearchLive 真打缺省后端（brave HTML）的联调测试（默认跳过；WEBSEARCH_LIVE=1 时运行）。
func TestSearchLive(t *testing.T) {
	if os.Getenv("WEBSEARCH_LIVE") == "" {
		t.Skip("set WEBSEARCH_LIVE=1 to run live search")
	}
	c := NewClient(Config{Enabled: true, Backend: "brave", MaxResults: 5, Timeout: 20 * time.Second})
	res, err := c.Search(context.Background(), "Go 1.24 new features", 5, nil, nil)
	if err != nil {
		t.Fatalf("live search: %v", err)
	}
	if len(res) == 0 {
		t.Fatal("live search returned no results")
	}
	for _, r := range res {
		t.Logf("%s | %s", r.Title, r.URL)
	}
}

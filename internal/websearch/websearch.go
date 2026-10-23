// Package websearch 为 /v1/messages 的 Anthropic web_search 服务端工具提供真实
// 检索能力。
//
// 背景：Anthropic 的 web_search 是**服务端工具**——检索由上游执行，客户端只渲染
// 结果。本网关的上游（CodeBuddy 逆向接口）没有该能力，因此由网关自己执行检索，
// 再把结果回填成 Anthropic 的 web_search_tool_result 内容块，并把摘要拼进出站
// prompt 让模型基于结果作答。
//
// 后端可插拔（web_search.backend）：
//   - brave      （缺省）search.brave.com HTML 抓取，无 key，国内直连可用；配了
//     web_search.api_key 时改走 Brave 官方 API；
//   - bing       www.bing.com 的 RSS 输出（&format=rss），无 key；实测结果质量差
//     （易返回无关推广），仅在其它后端不可用时兜底；
//   - duckduckgo html.duckduckgo.com HTML 抓取，无 key（易被 bot 挑战拦截）；
//   - searxng    自建/第三方 SearXNG JSON API（web_search.base_url）；
//   - tavily     Tavily Search API（web_search.api_key）。
//
// 出站代理：web_search.proxy 显式指定；为空时回落 HTTP(S)_PROXY 环境变量。
package websearch

import (
	"bytes"
	"context"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"html"
	"io"
	"log"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

// Result 一条检索结果（跨后端的统一形态）。
type Result struct {
	URL     string
	Title   string
	Snippet string
	PageAge string // 可空（ISO8601 或后端给出的相对时间）
}

// Config 检索配置（对应 config.json 的 web_search 段）。
type Config struct {
	Enabled    bool
	Backend    string
	BaseURL    string        // searxng
	APIKey     string        // tavily / brave
	MaxResults int           // 缺省 5
	Timeout    time.Duration // 缺省 10s
	Proxy      string        // 空 = 用环境代理
}

// Client 检索客户端（并发安全：http.Client 可复用）。
type Client struct {
	cfg  Config
	http *http.Client
}

// NewClient 构造客户端。cfg.MaxResults<=0 回落 5，Timeout<=0 回落 10s，
// Backend 空回落 "bing"。返回值恒非 nil（Enabled() 判定是否真正可用）。
func NewClient(cfg Config) *Client {
	if cfg.MaxResults <= 0 {
		cfg.MaxResults = 5
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 10 * time.Second
	}
	if strings.TrimSpace(cfg.Backend) == "" {
		cfg.Backend = "brave"
	}
	tr := &http.Transport{}
	if strings.TrimSpace(cfg.Proxy) != "" {
		if u, err := url.Parse(cfg.Proxy); err == nil {
			tr.Proxy = http.ProxyURL(u)
		} else {
			tr.Proxy = http.ProxyFromEnvironment
		}
	} else {
		tr.Proxy = http.ProxyFromEnvironment
	}
	return &Client{cfg: cfg, http: &http.Client{Timeout: cfg.Timeout, Transport: tr}}
}

// Enabled 报告是否启用检索。
func (c *Client) Enabled() bool { return c != nil && c.cfg.Enabled }

// Backend 返回当前后端名（日志/观测用）。
func (c *Client) Backend() string { return c.cfg.Backend }

// Search 执行一次检索，返回经域名过滤后的结果（最多 max 条）。
// max<=0 时用配置的 MaxResults。检索失败返回 error（调用方渲染为
// web_search_tool_result_error，不伪造结果）。
func (c *Client) Search(ctx context.Context, query string, max int, allowed, blocked []string) ([]Result, error) {
	if !c.Enabled() {
		return nil, errors.New("web search disabled")
	}
	if strings.TrimSpace(query) == "" {
		return nil, errors.New("empty query")
	}
	if max <= 0 {
		max = c.cfg.MaxResults
	}
	var (
		res []Result
		err error
	)
	switch strings.ToLower(c.cfg.Backend) {
	case "searxng":
		res, err = c.searchSearxng(ctx, query, max)
	case "tavily":
		res, err = c.searchTavily(ctx, query, max, allowed, blocked)
	case "brave":
		if c.cfg.APIKey != "" {
			res, err = c.searchBraveAPI(ctx, query, max)
		} else {
			res, err = c.searchBraveHTML(ctx, query, max)
		}
	case "duckduckgo", "ddg":
		res, err = c.searchDuckDuckGo(ctx, query, max)
	case "bing":
		res, err = c.searchBing(ctx, query, max)
	default:
		return nil, fmt.Errorf("unknown web search backend %q", c.cfg.Backend)
	}
	if err != nil {
		return nil, err
	}
	res = filterDomains(res, allowed, blocked)
	if len(res) > max {
		res = res[:max]
	}
	return res, nil
}

// ---------------------------------------------------------------------------
// JSON 后端
// ---------------------------------------------------------------------------

func (c *Client) searchSearxng(ctx context.Context, query string, max int) ([]Result, error) {
	base := strings.TrimRight(c.cfg.BaseURL, "/")
	if base == "" {
		return nil, errors.New("web_search.base_url is required for searxng")
	}
	u := base + "/search?format=json&q=" + url.QueryEscape(query)
	var out struct {
		Results []struct {
			URL           string `json:"url"`
			Title         string `json:"title"`
			Content       string `json:"content"`
			PublishedDate string `json:"publishedDate"`
		} `json:"results"`
		UnresponsiveEngines any `json:"unresponsive_engines"`
	}
	if err := c.getJSON(ctx, u, nil, &out); err != nil {
		return nil, err
	}
	// SearXNG 是聚合检索：多个引擎（bing + baidu + …）命中同一 URL 会重复出现。
	// 按 URL 去重后再截断 max，否则同一页占多个名额、等效结果面变窄。
	res := make([]Result, 0, len(out.Results))
	seen := make(map[string]bool, len(out.Results))
	for _, r := range out.Results {
		if r.URL == "" || seen[r.URL] {
			continue
		}
		seen[r.URL] = true
		res = append(res, Result{URL: r.URL, Title: r.Title, Snippet: r.Content, PageAge: r.PublishedDate})
		if len(res) >= max {
			break
		}
	}
	// 空结果 + 引擎全挂：SearXNG 此时仍回 200，静默会让「没搜到」和「引擎故障」
	// 无法区分。打一行 WARN 供排障（unresponsive_engines 是 [engine, error] 对的
	// 列表或字符串列表，直接 %v 原样带出）。
	if len(res) == 0 && out.UnresponsiveEngines != nil {
		log.Printf("WARN: [websearch] searxng: 0 results, unresponsive_engines=%v", out.UnresponsiveEngines)
	}
	return res, nil
}

func (c *Client) searchTavily(ctx context.Context, query string, max int, allowed, blocked []string) ([]Result, error) {
	if c.cfg.APIKey == "" {
		return nil, errors.New("web_search.api_key is required for tavily")
	}
	payload := map[string]any{
		"api_key":     c.cfg.APIKey,
		"query":       query,
		"max_results": max,
	}
	if len(allowed) > 0 {
		payload["include_domains"] = allowed
	}
	if len(blocked) > 0 {
		payload["exclude_domains"] = blocked
	}
	raw, _ := json.Marshal(payload)
	var out struct {
		Results []struct {
			URL           string `json:"url"`
			Title         string `json:"title"`
			Content       string `json:"content"`
			PublishedDate string `json:"published_date"`
		} `json:"results"`
	}
	if err := c.postJSON(ctx, "https://api.tavily.com/search", nil, raw, &out); err != nil {
		return nil, err
	}
	res := make([]Result, 0, len(out.Results))
	for _, r := range out.Results {
		res = append(res, Result{URL: r.URL, Title: r.Title, Snippet: r.Content, PageAge: r.PublishedDate})
	}
	return res, nil
}

func (c *Client) searchBraveAPI(ctx context.Context, query string, max int) ([]Result, error) {
	if c.cfg.APIKey == "" {
		return nil, errors.New("web_search.api_key is required for the Brave API")
	}
	u := "https://api.search.brave.com/res/v1/web/search?count=" + fmt.Sprint(max) + "&q=" + url.QueryEscape(query)
	hdr := map[string]string{
		"Accept":               "application/json",
		"X-Subscription-Token": c.cfg.APIKey,
	}
	var out struct {
		Web struct {
			Results []struct {
				URL         string `json:"url"`
				Title       string `json:"title"`
				Description string `json:"description"`
				Age         string `json:"age"`
			} `json:"results"`
		} `json:"web"`
	}
	if err := c.getJSON(ctx, u, hdr, &out); err != nil {
		return nil, err
	}
	res := make([]Result, 0, len(out.Web.Results))
	for _, r := range out.Web.Results {
		res = append(res, Result{URL: r.URL, Title: r.Title, Snippet: r.Description, PageAge: r.Age})
	}
	return res, nil
}

// ---------------------------------------------------------------------------
// HTML 抓取后端
// ---------------------------------------------------------------------------

var (
	ddgResultRe  = regexp.MustCompile(`(?is)<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>`)
	ddgSnippetRe = regexp.MustCompile(`(?is)<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>`)
	tagRe        = regexp.MustCompile(`(?is)<[^>]+>`)

	braveOpenRe  = regexp.MustCompile(`(?is)<div class="snippet[^"]*"[^>]*data-type="web"[^>]*>`)
	braveHrefRe  = regexp.MustCompile(`(?is)<a href="(https?://[^"]+)"`)
	braveTitleRe = regexp.MustCompile(`(?is)<div class="title search-snippet-title[^"]*">(.*?)</div>`)
	braveDescRe  = regexp.MustCompile(`(?is)<div class="content [^"]*line-clamp-dynamic[^"]*">(.*?)</div>`)
)

// bingRSSDoc Bing RSS 输出（&format=rss）的解析结构。
type bingRSSDoc struct {
	Channel struct {
		Items []struct {
			Title       string `xml:"title"`
			Link        string `xml:"link"`
			Description string `xml:"description"`
		} `xml:"item"`
	} `xml:"channel"`
}

func (c *Client) searchDuckDuckGo(ctx context.Context, query string, max int) ([]Result, error) {
	body, err := c.getHTML(ctx, "https://html.duckduckgo.com/html/?q="+url.QueryEscape(query), nil)
	if err != nil {
		return nil, err
	}
	return parseDDGHTML(body, max), nil
}

// searchBraveHTML 抓取 search.brave.com 的 HTML 结果页（无 key 后端）。
// 偶发连接被重置（EOF）→ 重试一次。
func (c *Client) searchBraveHTML(ctx context.Context, query string, max int) ([]Result, error) {
	hdr := map[string]string{
		"Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
		"User-Agent":      "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
	}
	u := "https://search.brave.com/search?q=" + url.QueryEscape(query)
	var body string
	var err error
	for attempt := 0; attempt < 2; attempt++ {
		if body, err = c.getHTML(ctx, u, hdr); err == nil {
			break
		}
	}
	if err != nil {
		return nil, err
	}
	return parseBraveHTML(body, max), nil
}

func (c *Client) searchBing(ctx context.Context, query string, max int) ([]Result, error) {
	hdr := map[string]string{
		"Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
		"User-Agent":      "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
	}
	u := "https://www.bing.com/search?format=rss&count=" + fmt.Sprint(max) + "&q=" + url.QueryEscape(query)
	body, err := c.getHTML(ctx, u, hdr)
	if err != nil {
		return nil, err
	}
	return parseBingRSS(body, max), nil
}

// parseBraveHTML 解析 search.brave.com 的 HTML 结果页。结果块为
// <div class="snippet" data-type="web">，块内首个外链 + title/description 两个 div。
// RE2 不支持环视，故用起始标签的位置切片划分块。
func parseBraveHTML(body string, max int) []Result {
	locs := braveOpenRe.FindAllStringIndex(body, -1)
	res := make([]Result, 0, len(locs))
	for i, loc := range locs {
		if len(res) >= max {
			break
		}
		end := len(body)
		if i+1 < len(locs) {
			end = locs[i+1][0]
		}
		block := body[loc[1]:end]
		hm := braveHrefRe.FindStringSubmatch(block)
		if len(hm) < 2 {
			continue
		}
		link := html.UnescapeString(hm[1])
		if !strings.HasPrefix(link, "http") {
			continue
		}
		title := ""
		if tm := braveTitleRe.FindStringSubmatch(block); len(tm) >= 2 {
			title = cleanHTML(tm[1])
		}
		snip := ""
		if dm := braveDescRe.FindStringSubmatch(block); len(dm) >= 2 {
			snip = cleanHTML(dm[1])
		}
		res = append(res, Result{URL: link, Title: title, Snippet: snip})
	}
	return res
}

// parseDDGHTML 解析 DuckDuckGo HTML 结果页。
func parseDDGHTML(body string, max int) []Result {
	links := ddgResultRe.FindAllStringSubmatch(body, -1)
	snips := ddgSnippetRe.FindAllStringSubmatch(body, -1)
	res := make([]Result, 0, len(links))
	for i, m := range links {
		if len(res) >= max {
			break
		}
		u := decodeDDGURL(m[1])
		if u == "" {
			continue
		}
		snip := ""
		if i < len(snips) {
			snip = cleanHTML(snips[i][1])
		}
		res = append(res, Result{URL: u, Title: cleanHTML(m[2]), Snippet: snip})
	}
	return res
}

// parseBingRSS 解析 Bing RSS 输出。描述字段可能含 HTML 片段，去标签后作为摘要。
func parseBingRSS(body string, max int) []Result {
	var doc bingRSSDoc
	if err := xml.Unmarshal([]byte(body), &doc); err != nil {
		return nil
	}
	res := make([]Result, 0, len(doc.Channel.Items))
	for _, it := range doc.Channel.Items {
		if len(res) >= max {
			break
		}
		link := strings.TrimSpace(html.UnescapeString(it.Link))
		if !strings.HasPrefix(link, "http") {
			continue
		}
		res = append(res, Result{
			URL:     link,
			Title:   strings.TrimSpace(html.UnescapeString(it.Title)),
			Snippet: cleanHTML(it.Description),
		})
	}
	return res
}

// ---------------------------------------------------------------------------
// HTTP 辅助
// ---------------------------------------------------------------------------

func (c *Client) getJSON(ctx context.Context, u string, hdr map[string]string, out any) error {
	body, err := c.getHTML(ctx, u, hdr)
	if err != nil {
		return err
	}
	if err := json.Unmarshal([]byte(body), out); err != nil {
		return fmt.Errorf("decode search response: %w", err)
	}
	return nil
}

func (c *Client) postJSON(ctx context.Context, u string, hdr map[string]string, payload []byte, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u, bytes.NewReader(payload))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("search backend HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("decode search response: %w", err)
	}
	return nil
}

func (c *Client) getHTML(ctx context.Context, u string, hdr map[string]string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Accept", "text/html,application/json;q=0.9,*/*;q=0.8")
	if _, ok := hdr["User-Agent"]; !ok {
		req.Header.Set("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
	}
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return "", err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("search backend HTTP %d", resp.StatusCode)
	}
	return string(raw), nil
}

// ---------------------------------------------------------------------------
// 解析/过滤辅助
// ---------------------------------------------------------------------------

// decodeDDGURL 从 DuckDuckGo 跳转链接里取出真实 URL（uddg 参数），取不到时返回原值。
func decodeDDGURL(href string) string {
	href = html.UnescapeString(href)
	if strings.HasPrefix(href, "//") {
		href = "https:" + href
	}
	u, err := url.Parse(href)
	if err != nil {
		return href
	}
	if v := u.Query().Get("uddg"); v != "" {
		return v
	}
	return href
}

// cleanHTML 去标签、解实体、压空白。
func cleanHTML(s string) string {
	s = tagRe.ReplaceAllString(s, "")
	s = html.UnescapeString(s)
	return strings.Join(strings.Fields(s), " ")
}

// filterDomains 按 allowed/blocked 域名后缀过滤。allowed 非空时只保留命中项。
func filterDomains(res []Result, allowed, blocked []string) []Result {
	if len(allowed) == 0 && len(blocked) == 0 {
		return res
	}
	out := make([]Result, 0, len(res))
	for _, r := range res {
		host := hostOf(r.URL)
		if host == "" {
			continue
		}
		if len(allowed) > 0 && !matchDomain(host, allowed) {
			continue
		}
		if len(blocked) > 0 && matchDomain(host, blocked) {
			continue
		}
		out = append(out, r)
	}
	return out
}

func hostOf(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	return strings.ToLower(u.Hostname())
}

// matchDomain 判断 host 是否命中域（含子域）之一。
func matchDomain(host string, domains []string) bool {
	for _, d := range domains {
		d = strings.ToLower(strings.TrimSpace(d))
		d = strings.TrimPrefix(d, ".")
		if d == "" {
			continue
		}
		if host == d || strings.HasSuffix(host, "."+d) {
			return true
		}
	}
	return false
}

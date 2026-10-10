package server

import (
	"fmt"
	"strings"
)

// ModelAlias 一条模型别名规则（config model_aliases 数组元素）。
//
//	From：客户端请求里出现的外部模型名（如 claude-sonnet-4-5 / gpt-5）。
//	To：网关实际支持的模型名，可带 "cn:"/"global:" realm 前缀，也可为裸名（默认 cn）。
type ModelAlias struct {
	From string `json:"from"`
	To   string `json:"to"`
}

// AliasTable 已校验的精确匹配别名表。nil 表 = 无别名（恒等映射）。
//
// 语义（v1）：
//   - 只做精确匹配（exact），不支持前缀/通配；未命中即原样透传。
//   - 匹配发生在 realm 前缀解析之前：命中后把整串替换为 To，再交给既有
//     resolveModel 解析 realm + 裸名。因此别名只决定"用哪个模型"，后续的选号、
//     账号策略、脱敏、提示词改写、出站传输全部沿用既有管线，不绕过任何安全处理。
//   - From 不得含 ':'：带 realm 前缀的模型名由网关自行解析，不走别名（避免与
//     realm 语法冲突）。
type AliasTable struct {
	m map[string]string
}

// NewAliasTable 校验并构建别名表。空 entries 返回 (nil, nil)：调用方按"无别名"处理，
// 保证未配置时行为与引入前完全一致。
//
// 校验（fail fast，错误可读、定位到具体条目）：
//   - from / to 为空；
//   - from 含 ':'（与 realm 前缀语法冲突）；
//   - from / to 含空白或 '*'（v1 只支持精确匹配，不接受通配写法）；
//   - from 重复定义（映射冲突）；
//   - to 解析后裸名为空（目标模型名非法）；
//   - to 本身是另一个 from（不支持链式别名，避免递归与歧义）。
//
// 注意：to 指向的模型是否"真实存在"由上游运行时判定（未知模型 → 上游错误码，
// 走既有错误分类与提示链路）；模型目录是动态拉取的，配置期无法离线穷举，故此处
// 只做格式校验，不做存在性校验。
func NewAliasTable(entries []ModelAlias) (*AliasTable, error) {
	if len(entries) == 0 {
		return nil, nil
	}
	m := make(map[string]string, len(entries))
	for i, e := range entries {
		from := strings.TrimSpace(e.From)
		to := strings.TrimSpace(e.To)
		if from == "" {
			return nil, fmt.Errorf("model_aliases[%d]: from 不能为空", i)
		}
		if to == "" {
			return nil, fmt.Errorf("model_aliases[%d] (from=%q): to 不能为空", i, e.From)
		}
		if strings.Contains(from, ":") {
			return nil, fmt.Errorf("model_aliases[%d]: from %q 不得含 ':'（realm 前缀由网关解析，别名只匹配外部名）", i, e.From)
		}
		if strings.ContainsAny(from, " \t\r\n*") {
			return nil, fmt.Errorf("model_aliases[%d]: from %q 含空白或通配符，v1 仅支持精确匹配", i, e.From)
		}
		if strings.ContainsAny(to, " \t\r\n*") {
			return nil, fmt.Errorf("model_aliases[%d] (from=%q): to %q 含空白或通配符", i, e.From, e.To)
		}
		if _, dup := m[from]; dup {
			return nil, fmt.Errorf("model_aliases: from %q 重复定义（映射冲突）", from)
		}
		if _, bare := resolveModel(to); bare == "" {
			return nil, fmt.Errorf("model_aliases[%d] (from=%q): to %q 解析后模型名为空", i, e.From, e.To)
		}
		m[from] = to
	}
	// 链式别名检查（单跳）：to 若本身是另一个 from 则拒绝。
	for from, to := range m {
		if _, isAlias := m[to]; isAlias {
			return nil, fmt.Errorf("model_aliases: %q 的目标 %q 本身是别名（不支持链式别名）", from, to)
		}
	}
	return &AliasTable{m: m}, nil
}

// Apply 返回别名映射后的模型名；未命中（或 nil 表）原样返回。
func (t *AliasTable) Apply(model string) string {
	if t == nil {
		return model
	}
	if to, ok := t.m[model]; ok {
		return to
	}
	return model
}

// Len 返回规则数（nil 表为 0），供日志/面板展示。
func (t *AliasTable) Len() int {
	if t == nil {
		return 0
	}
	return len(t.m)
}

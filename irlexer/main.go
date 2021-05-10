package main

import (
	"fmt"
	"strings"
	"encoding/json"
	"bufio"
	"regexp"
	"strconv"
	"os"
	"log"
	"os/exec"
	"sort"
	"github.com/ianlancetaylor/demangle"
	"github.com/llir/ll"
)

func splitIdentifier(identifier string) []string {
	res := []string {}
	splited := strings.Split(identifier, "_")
	for _, s := range splited {
		charType := 0
		lastCharType := 0
		if s == "" {
			continue
		}
		var builder strings.Builder
		for _, c := range s {
			lastCharType = charType
			if c >= 'A' && c <= 'Z' {
				charType = 1
			} else if c >= 'a' && c <= 'z' {
				charType = 2
			} else {
				charType = 3
			}
			if lastCharType != 0 && charType != lastCharType {
				st := builder.String()
				if lastCharType == 1 && charType == 2 {
					if len(st) > 1 {
						builder.Reset()
						token := strings.ToLower(st[:len(st)-1])
						token = strings.Trim(token, " ")
						if len(token) > 0 {
							res = append(res, token)
						}
						builder.WriteByte(st[len(st)-1])
					} else {
						builder.Reset()
						builder.WriteByte(st[len(st)-1])
					}
				} else {
					token := strings.Trim(strings.ToLower(st), " ")
					if len(token) > 0 {
						res = append(res, token)
					}
					builder.Reset()
				}
			}
			builder.WriteRune(c)
		}
		if builder.Len() != 0 {
			token := strings.Trim(strings.ToLower(builder.String()), " ")
			if len(token) > 0 {
				res = append(res, token)
			}
		}
	}
	return res
}

func splitSymbol(symbol string) ([]string, error) {
	ops := []string {
		"<<=", ">>=",
		"::", "++", "--", "&&", "||", "->", "<<", ">>", "<=", ">=", "==", "!=", "+=", "-=",
		"*=", "/=", "%=", "&=", "^=", "|=", "?", ",", "+", "-", "*", "/", "&", "|", "=",
		"^", "~", "!", "(", ")", "[", "]", ".", "{", "}", "<", ">", "%", "'", "\"", ":", ";",
	}
	tokens := []string {}
	pos := 0
	length := len(symbol)
	var builder strings.Builder
	for pos < length {
		if symbol[pos] == ' ' {
			if builder.Len() != 0 {
				tokens = append(tokens, splitIdentifier(builder.String())...)
				builder.Reset()
			}
			pos++
			continue
		}
		isOp := false
		for _, o := range ops {
			if pos + len(o) >= length {
				continue
			}
			if symbol[pos:pos+len(o)] == o {
				tokens = append(tokens, splitIdentifier(builder.String())...)
				builder.Reset()
				pos += len(o)
				tokens = append(tokens, o)
				isOp = true
				break
			}
		}
		if !isOp {
			builder.WriteByte(symbol[pos])
			pos++
		}
	}
	if builder.Len() != 0 {
		tokens = append(tokens, splitIdentifier(builder.String())...)
	}
	return tokens, nil
}

func processFuncSymbol(symbol string) []string {
	if strings.HasPrefix(symbol, "llvm.")  {
		splited := strings.Split(symbol, ".")
		return splited[1:len(splited)-1]
	}
	
	if pos := strings.LastIndexByte(symbol, '.'); pos != -1 {
		s := symbol[pos+1:len(symbol)]
		_, err := strconv.Atoi(s)
		if err == nil && pos == 1 && symbol[0] == 'f' {
			symbol = "f"
		} else {
			symbol = s
		}
	}
	symbol = demangle.Filter(symbol, demangle.NoParams, demangle.NoTemplateParams)
	if pos := strings.LastIndex(symbol, "::"); pos != -1 {
		symbol = symbol[pos+2:len(symbol)]
	}
	return splitIdentifier(symbol)
}

func removeTemplate(symbol string) string {
	depth := 0
	var builder strings.Builder
	for _, c := range symbol {
		if c == '<' {
			depth++
		} else if c == '>' {
			depth--
		} else if depth == 0 {
			builder.WriteRune(c)
		}
	}
	return builder.String()
}

func processTypeSymbol(symbol string) []string {
	symbol = removeTemplate(symbol)
	symbol = strings.TrimPrefix(symbol, "class.")
	symbol = strings.TrimPrefix(symbol, "struct.")
	if pos := strings.LastIndexByte(symbol, '.'); pos != -1 {
		symbol = symbol[0:pos]
	}
	if pos := strings.LastIndex(symbol, "::"); pos != -1 {
		symbol = symbol[pos+2:len(symbol)]
	}
	return splitIdentifier(symbol)
}

type Pair struct {
	word string
	num int
}
type Pairs []Pair
func (p Pairs) Len() int {
	return len(p)
}
func (p Pairs) Less(i, j int) bool {
	return p[i].num > p[j].num
}
func (p Pairs) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

func readDict(filename string) map[string]int {
	file, err := os.Open(filename)
	if err != nil {
		return nil
	}
	res := make(map[string]int)
	defer file.Close()
	reader := bufio.NewReader(file)
	for {
		b, err := reader.ReadBytes('\n')
		if err != nil {
			break
		}
		s := strings.Trim(string(b), "\n")
		splited := strings.Split(s, " ")
		word := splited[0]
		num, _ := strconv.Atoi(splited[1])
		res[word] = num
	}
	return res
}

func writeDict(filename string, dict map[string]int, minFreq int) {
	file, err := os.Create(filename)
	if err != nil {
		return
	}
	defer file.Close()
	var wordPairs Pairs
	for k, v := range dict {
		wordPairs = append(wordPairs, Pair {k, v,})
	}
	sort.Sort(wordPairs)
	w := bufio.NewWriter(file)
	for _, p := range wordPairs {
		if p.num >= minFreq {
			fmt.Fprintf(w, "%s %d\n", p.word, p.num)
		}
	}
	w.Flush()
}

func buildDictionary(filename string, totalDictFilename string, symbolDictFilename string, tokensFilename string, maxLen int, minFreq int) {
	file, err := os.Open(filename)
	if err != nil {
		return
	}
	defer file.Close()
	totalDict := make(map[string]int)
	symbolDict := make(map[string]int)
	reader := bufio.NewReader(file)
	var tokens []string
	for {
		b, err := reader.ReadBytes('\n')
		if err != nil {
			break
		}
		var insts []string
		err = json.Unmarshal(b, &insts)
		if err != nil {
			continue
		}
		if len(insts) > maxLen {
			continue
		}
		for _, inst := range insts {
			var lexer ll.Lexer
			lexer.Init(string(inst))
			for {
				token := lexer.Next()
				if token == ll.EOI {
					break
				}
				token_text := lexer.Text()
				if token == ll.GLOBAL_IDENT_TOK {
					symbols := processFuncSymbol(token_text[1:len(token_text)])
					tokens = append(tokens, symbols...)
					for _, t := range symbols {
						totalDict[t]++
						symbolDict[t]++
					}
				} else if token == ll.LOCAL_IDENT_TOK {
					var symbols []string = nil
					if strings.HasPrefix(token_text, "%\"") {
						symbols = processTypeSymbol(token_text[2:len(token_text)-1])
					} else if strings.HasPrefix(token_text, "%struct") || strings.HasPrefix(token_text, "%class") {
						symbols = processTypeSymbol(token_text[1:len(token_text)])
					} else {
						totalDict[token_text]++
					}
					if symbols != nil {
						tokens = append(tokens, symbols...)
						for _, t := range symbols {
							totalDict[t]++
							symbolDict[t]++
						}
					}
				} else {
					totalDict[token_text]++
				}
			} 
		}
	}
	tokensFile, err := os.Create(tokensFilename)
	if err != nil {
		return
	}
	defer tokensFile.Close()
	w := bufio.NewWriter(tokensFile)
	for _, t := range tokens {
		if symbolDict[t] >= minFreq {
			fmt.Fprintln(w, t)
		}
	}
	writeDict(totalDictFilename, totalDict, minFreq)
	writeDict(symbolDictFilename, symbolDict, minFreq)
}

func isNumeric(s string) bool {
	_, err := strconv.ParseFloat(s, 64)
	if err == nil {
		return true
	}
	if strings.HasPrefix(s, "0x") || strings.HasPrefix(s, "0X") {
		return true
	}
	return false
}

func processInst(instFilename string, totalDictFilename string, symbolDictFilename string, bpeProgram string,
    bpeFilename string, maxLen int) {
	file, err := os.Open(instFilename)
	if err != nil {
		return
	}
	defer file.Close()
	totalDict := readDict(totalDictFilename)
	//var symbolDict map[string]int = nil
	//if symbolDictFilename != "-" {
	//	symbolDict = readDict(symbolDictFilename)
	//}
	reader := bufio.NewReader(file)
	args := []string{"applybpe_stream", bpeFilename}
	if symbolDictFilename != "-" {
		args = append(args, symbolDictFilename)
	}
	bpeCmd := exec.Command(bpeProgram, args...)
	stdinPipe, err := bpeCmd.StdinPipe()
	if err != nil {
		log.Fatal(err)
		return
	}
	stdoutPipe, err := bpeCmd.StdoutPipe()
	if err != nil {
		log.Fatal(err)
		return
	}
	bpeReader := bufio.NewReader(stdoutPipe)
	bpeWriter := bufio.NewWriter(stdinPipe)
	err = bpeCmd.Start()
	if err != nil {
		log.Fatal(err)
		return
	}
	metadataRegexp, _ := regexp.Compile(", !insn\\.addr.*")
	for {
		b, err := reader.ReadBytes('\n')
		if err != nil {
			break
		}
		var insts []string
		err = json.Unmarshal(b, &insts)
		if err != nil {
			continue
		}
		if len(insts) > maxLen {
			continue
		}
		var res [][]string
		for _, inst := range insts {
			var lexer ll.Lexer
			lexer.Init(metadataRegexp.ReplaceAllString(string(inst), ""))
			var tokens []string
			isFunc := false
			isNum := false
			for {
				token := lexer.Next()
				if token == ll.EOI {
					break
				}
				var symbolsOrNumeric []string = nil
				token_text := lexer.Text()
				if token == ll.GLOBAL_IDENT_TOK {
					symbolsOrNumeric = processFuncSymbol(token_text[1:len(token_text)])
					isFunc = true
				} else if token == ll.LOCAL_IDENT_TOK {
					if strings.HasPrefix(token_text, "%\"") {
						symbolsOrNumeric = processTypeSymbol(token_text[2:len(token_text)-1])
					} else if strings.HasPrefix(token_text, "%struct") || strings.HasPrefix(token_text, "%class") {
						symbolsOrNumeric = processTypeSymbol(token_text[1:len(token_text)])
					} else {
						if totalDict[token_text] == 0 {
							tokens = append(tokens, "<unk>")
						} else {
							tokens = append(tokens, token_text)
						}
					}
				} else if isNumeric(token_text) {
					symbolsOrNumeric = []string {token_text}
					isNum = true
				} else if totalDict[token_text] == 0 {
					tokens = append(tokens, "<unk>")
				} else {
					tokens = append(tokens, token_text)
				}
				if symbolsOrNumeric != nil {
					var bpeSymbols []string
					for _, s := range symbolsOrNumeric {
						fmt.Fprintln(bpeWriter, s)
						bpeWriter.Flush()
						b, _ := bpeReader.ReadBytes('\n')
						s = strings.Trim(string(b), "\n")
						s = strings.ReplaceAll(s, "@@", "")
						bpeSymbols = append(bpeSymbols, strings.Split(s, " ")...)
					}
					if len(bpeSymbols) <= 8 {
						tokens = append(tokens, bpeSymbols...)
					} else {
						tokens = append(tokens, bpeSymbols[0:8]...)
						if isFunc {
							tokens = append(tokens, "</func>")
						} else if isNum {
							tokens = append(tokens, "</const>")
						} else {
							tokens = append(tokens, "</type>")
						}
					}
				}
			}
			res = append(res, tokens)
		}
		b, err = json.Marshal(res)
		if err != nil {
			continue
		}
		fmt.Println(string(b))
	}
}

func atoi(s string) int {
	i, _ := strconv.Atoi(s)
	return i
}

func main() {
	if os.Args[1] == "build-dict" {
		buildDictionary(os.Args[2], os.Args[3], os.Args[4], os.Args[5], atoi(os.Args[6]), atoi(os.Args[7]))
	} else if os.Args[1] == "process-inst" {
		processInst(os.Args[2], os.Args[3], os.Args[4], os.Args[5], os.Args[6], atoi(os.Args[7]))
	}
}

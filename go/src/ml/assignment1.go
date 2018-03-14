package main

import (
  "bufio"
  //"io"
  //"io/ioutil"
	"encoding/csv"
	"fmt"
	//"log"
	//"strings"
  "os"
)
func check(e error) {
  if e != nil {
    panic(e)
  }
}
func main() {
  f, err := os.Open("ex1data1.txt")
  check(err)
	//in := `ex1data1.tx`
	r := csv.NewReader(bufio.NewReader(f))
  r.Comma = ','
	record, err := r.ReadAll()
	check(err)
  fmt.Println(record)
  fmt.Println(len(record))

//  for value := range record {
//    fmt.Printf("  %v\n", record[value])
//  }
	//fmt.Print(records)
}

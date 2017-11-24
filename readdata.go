package main

import (
	"github.com/petar/GoMNIST"
	"log"
	"fmt"
)


// This is just a playgrund to get the feeling of the data format
func main () {
	train, test, err := GoMNIST.Load("./data")
	if err != nil {
		log.Fatal(err)
	}


	sweeper := train.Sweep()

	sum := 0
	for sum < 10 {
		image, label, present := sweeper.Next()
		fmt.Println(len(image))
		fmt.Println(label)
		fmt.Println(present)
		sum++
	}

	sweeper = test.Sweep()

	sum = 0
	for sum < 10 {
		image, label, present := sweeper.Next()
		fmt.Println(len(image))
		fmt.Println(label)
		fmt.Println(present)
		sum++
	}
}


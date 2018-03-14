package main

import(
  "fmt"
  "github.com/gonum/matrix/mat64"
)

func main() {

    // initialize a 3 element float64 slice
    dx := make( []float64, 3 )

    // set the elements
    dx[0] = 2
    dx[1] = 2
    dx[2] = 3

    // pass the slice dx as data to the matrix x
    x := mat64.NewDense( 3, 1, dx )

    // alternatively, create the matrix y by
    // inserting the data directly as an argument
    y := mat64.NewDense( 3, 1, []float64{1, 4, 5})

    // create an empty matrix for the addition
    z := mat64.NewDense( 3, 1, []float64{0, 0, 0})
    c := mat64.NewDense(1,1,[]float64{0})
    d := mat64.NewDense(3,3,[]float64{0,0,0,0,0,0,0,0,0})
    // perform the addition
    z.Add( x, y )
    c.Mul(x.T(),y)
    d.Mul(x,y.T())
    // print the output
    fmt.Printf( "%f %f %f\n", z.At(0,0), z.At(1,0), z.At(2,0) )
    fmt.Printf(" mult = %f \n", c.At(0,0))
    fmt.Print("mul2\n")
    fmt.Printf( "%f %f %f\n", d.At(0,0), d.At(0,1), d.At(0,2) )
    fmt.Printf( "%f %f %f\n", d.At(1,0), d.At(1,1), d.At(1,2) )
    fmt.Printf( "%f %f %f\n", d.At(2,0), d.At(2,1), d.At(2,2) )

}

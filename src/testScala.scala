import java.nio.file.{Files, Paths}
object testScala {
  def main(args:Array[String]): Unit =
  {
    println("hello")
    val byteArray = Files.readAllBytes(Paths.get("/home/kingo/Workshop/dataset/t10k-labels-idx1-ubyte"))
    println(byteArray(2))
  }

}

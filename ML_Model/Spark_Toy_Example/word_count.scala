import org.apache.spark.SparkContext._
import org.apache.spark._

object WordCount {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("WordCount")
        val sc = new SparkContext(conf)

        rdd = sc.textFile("shakespeare.txt")
        words = rdd.flatMap(lambda x: x.split())
        words.count()
        word_counts = words.map(;ambda x: (x, 1))
        result = word_counts.map(lambda x: (x[1], x[0])).sortByKey(False)
        result.take(5)
    }

}
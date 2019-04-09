# day1 

```scala
/*
定义变量
object 中的o一定要小写 和Java的Object 是不一样
object创建的类是单利类
 */
object ValAndVar {
  //入口方法def开头的都是方法 main是方法名 () 是参数列表 args 参数名 Array[String]
  // : 后面 Unit --> 数据类型  就是Java (void) 没有返回值
  // = { 方法体}
    def main(args: Array[String]): Unit = {
    //在Scala中不需要显式的声明数据类型,因为Scala会根据值自动推断
    //定义变量需要使用到两个关键字 val 和 var
    //语法:  var或val 变量名(:数据类型) = 值
    //ps:()中的是可以省略的
    //val 修饰的变量是不可以被改变的 相当于java中 final
    val name = "小明"
   // name = "小红"
   //var 修饰的变量是可以修改值的
    var age = 18
   // age = 20

    //同时声明多个变量
    val name1,name2 = "111"
    val num1,num2 = 100

    //在创建变量的时候是不显式的使用数据类型的,但是scala中是存在着基本数据类型的
    //Byte,Char,Short,Int,Long,Float,Double,Boolean,这些都是类
    //这些类提提供了一些方法方便我们来使用,并且Scala会负责基本数据类型和引用类型的转换操作

   //ps:在声明变量的时候scala中建议使用val来声明变量

     /*
      八种基本数据类型在Scala提供了一个加强类Rich基本数据类型
      例如 RichInt --> Int的加强类
      */

    /*
      Scala中运算符和java中基本上没有什么区别,只不过
      1,Scala中没有 ++ 和  --
      2.常用的 + - * /% > < >= <= == != && || ! 都是和java一样
      3.Scala基本数据类型将运算符写成了方法,所以可以用过 对象.运算符(值) 进行计算
      */
     val sum = 1+1
     val sum1 = 1.+(1)

      //Unit表示无值,和void像是,表示这个你方法返回值

  }
}

```



```scala
/**
  * 1.scala中有while和do while基本语法和java是相同也即是说java怎么写Scala就怎么写 ,需要注意的是这里没有++ --
  * 所以自增或自减需要使用+= 或 -= 完成
  * 2.Scala中没有标准的for循环
  * 标准for --> for(循环变量赋初值;循环条件;循环变量的自增或自减){循环体}
  * 若需要使用这种类似的for只能用while代替
  * Scala中有类似于java中的增强for循环但是完全语法不一样
  * 3.scala中没有跳出循环语句
  *    虽然Scala中没有提供类似java中的break语句,但是可以使用如下三种方式完成循环停止
  *    作业:
  *    使用boolean类型变量 ,return,Breaks的break函数来停止
  *  使用频率较高的是while和for 用for局居多
  *
  */
object LoopDemo {
  def main(args: Array[String]): Unit = {
//     var i = 1
//     while(i < 10){
//       println(i)
//       i+=1
//     }
    //定义数组数组元素时1-6
    //数组是有可变和不可变之分
    val array = Array(1,2,3,4,5,6)
    //遍历数组并打印
    //类似于java的增强for循环
    //java中的增强for循环
    /*
      for(数据类型 变量名 : 集合名(数组名)){
          变量就会获得数组中每一个元素
      }
     */
    //scala 中的for循环  <- 指向
    for(ele<-array){
      println(ele)
    }
    //通过下标获取当前数组中的元素
    //to方法(前后都包括)  until方法(不包括结尾)
    /**
      * Range(1, 2, 3, 4, 5)
      * --------------------------风骚的分割线--------------------------
      * Range(1, 2, 3, 4)
      */
    val  r1  = 1 to 5
    println(r1)
    println("--------------------------风骚的分割线--------------------------")
    val  r2  = 1 until 5
    println(r2)

    for(i <- 0 to 5){//包含结尾
      println(array(i))//取出数组中的元素
    }
    for(i <- 0 until 6){//不包含结尾
      println(array(i))//取出数组中的元素
    }

    //for循环中可以添加守卫 -->添加条件
    //if即使守卫
    for(e <- array if e%2 == 0){
         println(e)
    }
    //for循环的嵌套
     for(i <- 1 to 3){
        for (j <- 1 to 3){
          if(i!=j){
            print((10*i+j)+" ")
          }
        }
       println()
     }
    //高级for循环(for循环嵌套)
    for(i <- 1 to 3;j <- 1 to 3 if i!=j){
      print((10*i+j)+" ")
    }
    println()

    //for循环的推导式,加入yield关键字,该循环会构建一个新的集合或数组,每次迭代会生成集合的一个新值

    var res  =  for(i<- 1 to 10) yield i
    println(res)

    //foreach  filter  map 函数使用和含义

    val arr = Array(1,2,3,4,5,6)
    //filter一个过滤器,根据传入的条件将数据过滤出来
      val arr1: Array[Int] = arr.filter(x => x%2 == 0)
      println(arr1.toBuffer)
      //map把数组中每一个元素都取出来的到一个全新数组
      val arr2: Array[Int] = arr.map(x => x)
      println(arr2.toBuffer)
      //foreach数组中的元素取出来并打印 (无返回值)
       arr.foreach(x => println(x))


  }
}

```



```scala
/**
  * Scala中的格式化输出
  */
object PrintDemo {
  def main(args: Array[String]): Unit = {
      val name = "千锋教育"
      val pirce = 30
      val url = "www.qfedu.com"
      //输出打印
      //1.普通版本 带ln的就是换行 不带就是不换行
      //可以向java一样进行字符串片接输出
      println("name="+name,"pirce="+pirce,"url="+url)
      //print中有一个带f后缀方法,类似于C语言的打印方法 printf
     // Java中的String --> format 格式控制符 ,对应的值
    // 第一个参数是格式化控制符,对应的值
      printf("名字:%s 市值:%d 网址:%s",name,pirce,url)//但是这个方法没有换行
       println()
      println(f"$name $pirce $url")
       //可以使用s来处理字符串中直接使用变量
      println(s"name=$name")
    //使用s处理字符串的是可以使用 ${}来完成一个表达式并进行计算
      println(s"1+1=${1+1}")
      println("1+1="+(1+1))
     //获取控制台上的数据
     val scan = readLine()
      println(scan)
  }
}

```

```scala
/**
  * 条件表达式
  * 1.if表达式的定义和java中的是基本相同的也是一个boolean类型的表达式,if或else中最后一个行语句可以使返回值
  *  例如 val age = 30
  *   if(age > 18) 1 else 0
  * val age = 30
  * // 这样做就相当于java当中的一个运算符 三目? :
  * val newAge = if(age > 18) 1 else 0
  * //上面的另外一种写法
  * var i = -1;if(age>18)i=1 else i=0
  * //也可以使用基本的形式来完成
  * // if(age>18) println("1") else println("0")
  * 2.if表达式可以推断类型,若if表达式有值或是if-else表达式有值,会自动推断类型
  * val age = 30
  * val newAge = if(age>18) 1 else 0 --> 得到结果的时候相当于 val newAge:Int = ?
  * //Scala中会自动推断类型,但是若存在多个不同的数据类型是,Scala会推断他们共有数据类型即父类Any
  * val newAge = if(age>18)"String" else 0
  * 若使用一个单独的if后面没有else
  * 例如 val age = 30
  * val newAge = if(age>18) "String" 此时scala会认为else是默认存在并且数据类型是Unit
  * if(age>18) "String" == if(age>18) "String" else () --> ()代表空什么也没有
  *
  *关系运算符和逻辑运算符的引用和Java是一样的 可以直接使用或是在if中使用
  *
  * 语句结束符合块表达
  * 1.默认scala是不需使用语句结束符的不需要使用;来分割
  * 2.一行有多条语句,着必须使用分割符分开,也即是[;]
  * 例如 var a,b,c=0;if(a<10){b=b+1;c=c+1}
  * 通常可以这样写
  * var a,b,c=0
  * if(a<10){
  * b=b+1
  * c=c+1
  * }
  * if中若是有多行语句建议使用{}
  * 3.{} 就是块表达式 在当前代码块中有多条语句,最后一条语句值是整个块表达式的返回值
  * var a = 2
  * var b = 1
  * var c = 1
  * val d = if (a < 10){
  * b = b+1
  * c = c+1
  * b
  * }
  * println(d)
  * scala中没有switch-case,提供了一个比他更强大的应用匹配模式
  */
object IfDemo {
  def main(args: Array[String]): Unit = {
    //if分支的应用
    val faceValue  = 99
    val rs  = if(faceValue > 90) "帅的不要不要的" else "瘪犊子玩意儿..."
    println(rs)
    val i = 8
    val rs1 = if (i > 8) i
    println(rs1)
    val rs2 = if (i > 8) i else "前面是一个整数这里是一个字符串rs2的数据类型是什么?"
    println(rs2)
    //if else if 的语句较多那么会使用{}来完成
    val score = 76

    val rs3 = {
      if (score > 60 && score < 70) "及格"
      else if (score >= 70 && score < 80) "良好"
      else "优秀"
    }
    println(rs3)

    //ps:
    var  rs4 = ""
    if (score > 60 && score < 70) {
      rs4 ="及格"
    } else if (score >= 70 && score < 80) {
      rs4 = "良好"
    } else {
      rs4 = "优秀"
    }
   println(rs4)

  }
}

```

# day2

```scala
/**
  *在java中是不区分方法和函数,既然是面向对象语言,统称为方法
  * 访问权限修饰符 [其他修饰符] 返回值类型 方法名(参数列表){ 方法体 }
  *
  * 在Scala中是严格区分方法和函数的
  * ps:scala函数可以想成是java中lambda表达式,有很大的区别
  * 语法:
  * def 方法名(参数列表): 返回值类型 = {
  *      方法体
  * }
  * ps:方法的返回值类型可以不写,编译器可以自动推断出返回值类型,但是对于递归方法,那么必须指定返回值
  *    若需要返回值的时候需要在方法的后面添加 =
  *    若不需要返回值直接实现即可 省略等号和返回值类型
  */
object MethodAndFunction {
  //标准方法  方法签名 sum(a:Int,b:Int):Int
  def sum(a:Int,b:Int):Int ={
     //scala中遵循最后一句即可以返回的原理可以省略不写return 就能返回
    a+b
  }
  //简化1:scala是可以自动推断返回值类型
  def sum1(a:Int,b:Int) ={
    //scala中遵循最后一句即可以返回的原理可以省略不写return 就能返回
    a+b
  }
  //简化2:Scala可以将代码写成一句 可以将方法简化一句
  //若需要返回值使用 = 不需要返回值 省略不写
  def sum2(a:Int,b:Int){println(a+b)}
  //简化3:返回值可以省略,大括号可以省略 但是 = 不能省略
  def sum3(a:Int,b:Int)= a+b
  //定义一个方法传入一个参数 ,打印九九成表
  def printlns(num:Int): Unit ={
    for(i <- 1 to num ;j <- 1 to i ){
      if(i == j) println(j+"*"+i+"="+(j*i)) else print(j+"*"+i+"="+(j*i)+"\t")
    }
  }
  //没有参数和没有返回值的方法
  //标准
  def sayHello():Unit={
    println("哈哈哈哈哈")
  }
  //简化1:
  def sayHello1() = println("哈哈哈")

  //最简
  def sayHello2 = println("哈哈哈")

   //scala的形式写斐波那契数列 方法版本
   // 1  1  2  3  5  8 ....
    //输入 3 -- > 2
   //递归方法 不能使用scala自身推断机制,必须写出返回值类型
    def fab(n:Int):Int = {
      if (n == 1 || n == 2) 1
      else fab(n-1)+fab(n-2)
   }


  def main(args: Array[String]): Unit = {
//    println(sum(1,5))
//    println(sum1(1,5))
//    sum2(1,5)

    //函数的声明
    //一般函数是声明在方法体的内部或作为参数进行传递使用
    //方法和函数的区别:函数可以做值以参数的形式传递到方体中,也即是说定义方法的时候,方法的参数可以使一个函数类型
    //ps:在定义方法时方法的形参列表是一个函数 或 返回值是一个函数
    //语法: val 函数的名字 = (参数列表) => { 函数体 }
    val f1 = (x: Int,y: Int) => { x * y}
    //通过解释器可以得知当前函数的类型
    //f1: (Int, Int) => Int  = <function2>
    // f1 是函数的名字 : (参数的数据类型并且有几个参数就有几个数据类型) => 函数的返回值是什么
    //function2 先忽略
    //调用函数
    //第一种直接调用和方一样 第二种作为参数传递
    println(f1(2,3))
    val arr = Array(1,2,3)
    //花括号可以省略
    val p = (x:Int) => x %2 ==0
    arr.filter(p)

    //没有参数没有返回值
    val f2 = () =>  println("1")
    f2()

    //匿名函数 作为方法的参数传递时使用只会只用一次
    //(x:Int,y:Int) => x * y

    //函数的另外一种定义
    //val 函数名:(参数的数据类型) => 返回值类型 ={(参数名) => 方法体}

    val f3:(Int,Int)=>Int = {(x,y) => x*y}
    val f4:(Int)=>Int = i=>i
    val f5:()=>Unit = () => println("1")

    //如何将方法转变为函数
    //ps:在系统提供的方法方法中有参数作为参数的数据,此时既可以传入函数也可以传入方法(自动将方法转换为函数)
    //手动转变方式 方法名[有空格]下划线
    val f6 = sum _

    //可变参数(函数不行,方法行)
    //可变参数需要在参数的最后一个之后一位,并不能有多个
    def month(a:String*) = {
      for(i<-a){
        println(i)
      }
    }
    month("1","2","3","4")

    //参数带有默认值
    def add(a:Int = 1 , b :Int = 7):Int ={
       println(a)
       println(b)
       a+b
    }
   //使用默认值
    println(add())
    println(add(1,2)) //修改值
    println(add(b = 3)) // 直接加变量名即可
  }
}
```

```scala
object NineLoop extends App {
    for(i <- 1 to 9){
      for(j <- 1 to i){
        print(j+"*"+i+"="+(j*i)+"\t")
      }
      println()
    }
    for(i <- 1 to 9 ; j <- 1 to i){
         if(i == j){
           println(j+"*"+i+"="+(j*i))
         }else{
           print(j+"*"+i+"="+(j*i)+"\t")
         }
    }

}
```

```scala
import scala.collection.mutable

/**
  * 集合之数组
  * scala中分可变和不可变数组
  * 定长和变长的数组 --> 把这个理解为是Java中的字符串
  * java中的字符串分为两种 String 或 (StringBuffer或StringBuilder)
  * String str = "abc"
  * String str1 = str.concat("d")
  * 在字符串池中 abcd 是一个字符串 abc 也是一个字符串
  * StringBuffer sbr = new  StringBuffer("abc")
  *   sbr.append("d")
  * 等于操作了同一个字符串相同 abc 变成了 abcd 只有一个字符串
  *
  * 在scala不导入包的情况下默认就是不可变 immutable
  * 可变数组导包mutable
  * import  scala.collection.mutable.ArrayBuffer 可变数组
  */
import  scala.collection.mutable.ArrayBuffer
object ArrayDemo {
  def main(args: Array[String]): Unit = {
     //定义数组
     var arr:Array[Int] = new Array[Int](3)
     //简化
     var arr1 = new Array[Int](3)

     val arr2 = Array(1,2,3)
     //操作当前数组,下标
      arr2(0) = 100
     println(arr2.toBuffer)
     // reverse翻转
    //排序默认是升序,降序先升序排序后翻转
    println(arr2.reverse.toBuffer)
    //排序sorted 升序排序 若想降序需要进行翻转
//    val sortedArray: Array[Int] = arr2.sorted
//    val reverseArray: Array[Int] = sortedArray.reverse
//    val buffer: mutable.Buffer[Int] = reverseArray.toBuffer
    println(arr2.sorted.reverse.toBuffer)

    //可变数组
     //创建数组
     val arr3  = new ArrayBuffer[Int](4)
     //向数组中添加元素
     arr3 += (1,2,3,4,5,6,7,8)
      println(arr3)
     //将一个不可变数组添加到可数组中
    arr3 ++= arr2
    println(arr3)
    //删除指定位置的元素(指定位置) 删除对应元素的值
//    val i: Int = arr3.remove(0)
//    println(i)
//    //删除指定位置,删除元素的个数(包含当前删除的位置)
//    arr3.remove(0,2)
//    println(arr3)
    //删除数组中对应的元素,若数组存在相同的元素 删除第一个元素
//     arr3 -= 2
//    println(arr3)
    //插入数据 第一个参数是开始的位置,插入多少个元素(可变参数)
    arr3.insert(0,-1,-2,-3)
   println(arr3)

    //sortwith 根据需求(传入的函数来决定如何排序)进行排序
    //sorted 直接升序排序,若想降序reverse
    /*
    sorted默认是升序并且会返回一个排序后的新数组,若想降序reverse
    可以自定升序或降序sortwith 参数是一个函数这个函数需要有两个参数进行比较返回的是一个boolean类型的值
     需要升序 < 号  降序 > 号
     */
//    val f1 = (x:Int,y:Int) =>  x < y
//    val newArray: ArrayBuffer[Int] = arr3.sortWith(f1)
 //   val newArray: ArrayBuffer[Int] = arr3.sortWith((x,y) =>  x < y)
    val newArray: ArrayBuffer[Int] = arr3.sortWith(_ < _)
     println(newArray)
    //获取数组长度
    println(arr3.length)


  }

}

/*
  数据清洗
 */
object ArrayDemo2 {
  def main(args: Array[String]): Unit = {
    //定义一个数组
    val words = Array("hello tom hello xiaoming hello jerry", "hello Hatano")
    //Array(hello, tom ,hello ,xiaoming ,hello ,jerry ,hello ,Hatano)
    //  map方法这个方法可以遍历数组中元素,通过自己定义的函数可以决定元素如何处理
    //取出数组中的元素将元素拆分得到了一个全新的数这个数组中存储的元素时一个数组
    //如何将数组中存储的数组取出来
    val wdArray: Array[Array[String]] = words.map(wd => wd.split(" "))
    //对数组中存储数组的这种形式提供一个扁平化操作
    //ps:通过当前flatten这个方法将复杂Array中的数据进行逐一的取出并赋值给一个全新的数组
    val flattenArray: Array[String] = wdArray.flatten
    println(flattenArray.toBuffer)

    //融合上面两个方法于一身即可以遍历有可以扁平化操作
    //先进行map操作然后在进行flatten操作
    val flatArray: Array[String] = words.flatMap(wd => wd.split(" "))
     println(flatArray.toBuffer)

    //遍历数组打印
     flatArray.foreach(println)

  }
}
```

```scala
/**
  * 会使用元组代替Map来使用
  * Scala中的元组是一个固定数量的组合,本体可以做一个参数传递
  * 元组可以容纳不通过类型的数据,但是它是不可变
  * 例子:学习java时候,需要方法返回两个值,数组,集合和自定义类
  *    在scala中只需使用一个元组类型即可
  *    Scala的 Tuple1  元组是有上限的, scala中最多只能有22个,若数据过大建议使用的即使集合了
  */
object TupleDemo {
  def main(args: Array[String]): Unit = {
    //定义元组
      val t = (1,"hello",true)
      ///或者
    //这个比较特殊 就是根据Tuple后面的数字决定数据的获取个数
      val tuple1 = new Tuple1(1)
      val tuple2 = new Tuple2(1,"hello")
        val tuple3 = new Tuple3(1,"hello",true)
    //元组有一个类似于Array一样的下标但是元组是 从1开始 逐渐递增
      println(tuple1._1)
      println(tuple2._2)

     val tuple4,(x,y,z) = (1,2,3)
     println(x)

  }

}

```

```scala
/**
  * 集合之List集合 可变List和 不可变List
  * mutable(可变)  immutable(不可变)
  * scala中要表示一个空集合 使用 Nil
  */
object ListDemo {
  def main(args: Array[String]): Unit = {
    // 不可变List
         val list1 = List(1,2,3)
      // :: 操作符将给定头或尾创建一个新的集合
        println(list1)
        val list2 = 0 :: list1
      //这个运算符是右结合   0 :: 9 :: 5 :: list1 --> 0 :: (9 :: (5 :: list1))
         println(list2)
        val list3 = list1.::(0)
        val list4 = 0 +: list1
        //尾结合
        val list5 = list1 :+ 0
        val list6 = List(4,5,6)
       //将连个list合并成一样全新的list
         val list7 = list1 ++ list6
         //将list1 插入到list6的前面的到一个新的集合
        val list8 = list6.:::(list1)
    //可变List
    import  scala.collection.mutable.ListBuffer
    //创建一个可变集合
    val lst0 = ListBuffer[Int](1,2,3)
    //空集合使用new  ListBuffer[Int] 或 ListBuffer[Int]()
    val lst1 =  ListBuffer[Int]()
    //添加数据
    lst1 += 4
    lst1.append(1,2,3,4,5)
    //将集合添加到另一个集合
    lst0 ++= lst1
    //全新的集合
    val lst2 = lst0 ++ lst1
  }
}
/**
  * set集合也有可变和不可变之分
  */

object SetDemo {
  def main(args: Array[String]): Unit = {
    //import scala.collection.immutable.HashSet
    //set集合是排重的
//     val set1 = new HashSet[Int]()
//     val set11 = HashSet[Int](12)
//     val set2 = set1 + 4
    import scala.collection.mutable.HashSet
    //可变
    val set3 = new  HashSet[Int]()
    set3 += 3
    set3.add(4)
  }
}

/**
  * Map也是可变和不可变 默认使用的是HashMap
  */
object MapDemo {
  def main(args: Array[String]): Unit = {
//    //创建Map
//     val map1 = Map(("xiaoming",18),("xiaohong",20),("xiaoli",30))
//      val map2 = Map("xiaoming"->18,"xiaohong"->20,"xiaoli"->30)
//    //获取对应value的值 通过key取出value
//    //scala中的map可以通过key获取value,但是key不存在直接崩溃
//     //val value = map1("dsfhkjasdhfkjasdhf")
//    //println(value)
//    //key获取value
//
//    /*
//     为了避免崩溃,可以使用系统提供的get方法参数是key获取value
//     若key存在会返回对应value 否则返回None
//     get方法的返回值是一个Option
//     Option[Int] --> 会尽力返回一个Int类型给定,没有那就给你None
//     Option有两个子类
//     一个是Some
//     一个是None
//     */
//     val value2: Option[Int] = map1.get("xiaoming")
//     println(value2)
//    val value3: Option[Int] = map1.get("asdjfilasdf")
//    println(value3)
//
//    //需要判断这个值是否是None
//    if(!value2.isEmpty){
//      //通过getOrElse获取当前这个值,若取值失败就会返回默认值
//      //就是()中的参数
//      val i: Int = value2.getOrElse(0)
//      println(i)
//    }
//    //通过key获取vlaue或key不存在直接返回第二个参数的默认值
//    val value4: Int = map1.getOrElse("xiaoming",0)

    //可变Map
    import  scala.collection.mutable.Map
    //创建Map
    val map1 = Map(("xiaoming",18),("xiaohong",20),("xiaoli",30))
    //修改值
     map1("xiaoming") = 19
     map1 += (("hadoop",10000))
     map1.put("storm",222)
    //获取数据就是get 或是 getOrElse
    //删除map中的数据
      map1 -= "hadoop"
      map1.remove("storm")
    //遍历map
    for((key,value) <- map1){
        println(key + " " + value)
    }
    //keySet -->获取所有key  values -->获取所有value
    for(key <- map1.keys) println(key)
    //生成一个map并翻转key和value
    val map2 = for((k,v)<-map1)yield(v,k)

  }
}

```

# day3

```scala
//scala中用到言延迟加载
object LazyDeme {
  def init():String = {
    println("call init()")
    return "最后的"
  }

  def main(args: Array[String]): Unit = {
    //并没有使用Lazy
//    val p = init()
//    println("after init()")
//    println(p)
    //使用Lazy作为变量的修饰
       lazy val p = init()
        println("after init()")
        println(p)
  }
}

```

```scala
import scala.collection.{SortedMap, mutable}

/*
集合可变和不可变 List  Set  Map  数组 --> immutable 不可变  mutable 可变
set-->HashSet  排重  无序
map-->Map(HashMap) key排重 但是无序
集合的可变和不可变根val 和var 无关的

 */
object MapDemo {
  def main(args: Array[String]): Unit = {
    //根据key来进行自动排序 升序
    val map1 = SortedMap((3,"小明"),(1,"小红"),(2,"小黄"))
    println(map1)

    //插入的顺序是什么 打印的顺序就是什么
    val map3 = mutable.LinkedHashMap[String,Int]()
    map3("yy1") = 20
    map3("yy3") = 20
    map3("yy2") = 20
    map3("yy4") = 20
    println(map3)

    //将数组转换为Map 数组中的元素必须是key-value形似
    val arr: Array[(String, String)] = Array(("key1","value1"),("key2","value2"),("key2","value2"))
    val map: Map[String, String] = arr.toMap

    val arr1 = Array("tom","jerry","kitty","kkk")
    val arr2 = Array(3,2,1)
    //可以将两个数组汇总的数据写成对象的key-value形式 这种方式成为(拉链)
    //ps:做两个数组的拉链操作若其中某一个数组元素多余另外一个数组,多余的元素会被自动删除
     val array: Array[(String, Int)] = arr1.zip(arr2)
    //简化 arr1 zip arr2
    val map2: Map[String, Int] = array.toMap
       println(map2)

  }

}
```

```scala

object Lianxi {
  def main(args: Array[String]): Unit = {
    //  1.创建一个List集合 集合中存储着一些数字
    val list1 = List(3,1,2,7,5,9,4,6,8)
    //将list1中的每一个元素乘以2生成一个新的集合
    val list2 = list1.map(_ * 2)
    println(list2)
    //list1中的偶数取出来生成一个新的集合
    val list3: List[Int] = list1.filter(_ % 2 == 0)
     println(list3)
    //将list1排序后生成一个新的集合
    //sorted默认升序 翻转reverse降序 自定义 sortWith自定义
    val list4: List[Int] = list1.sorted
    println(list4)
    //分组每4个元素分为一组 参数是一组中的元素
    val iterator: Iterator[List[Int]] = list1.grouped(4)
    println(iterator)
    //Iterator转换成List
    val list5: List[List[Int]] = iterator.toList
      println(list5)
    //将集合中的数据扁平化
      val list6: List[Int] = list5.flatten
    println(list6)

    //有一个集合
    val lines = List("hello ni hao a","Hello wo hen hao")
    //将集合修改为  --> List(hello, ni, hao,a,Hello ,wo ,hen ,hao)
    //方法一
    val words1 = lines.map(_.split(" ")).flatten
    //方法二
    val words2 = lines.flatMap(_.split(" "))
  }
}
```

```scala
/**
  * scala版本的wordCount
  */
object WordCount extends  App {
  val lines = List("hello ni hao a","hello wo hen hao")
  //1 先将数据压平
  val words: List[String] = lines.flatMap(_.split(" "))
  //List(hello, ni, hao, a, hello, wo, hen, hao)
  //将list中的数据进行处理(将每一个单词添加1 方便计数)
  //List((hello,1), (ni,1), (hao,1), (a,1), (hello,1), (wo,1), (hen,1), (hao,1))
   val tuples: List[(String, Int)] = words.map((_,1))
  /**
    * Map(hen -> List((hen,1)), a -> List((a,1)), ni -> List((ni,1)), wo -> List((wo,1)),
    * hao -> List((hao,1), (hao,1)), hello -> List((hello,1), (hello,1)))
    */
    //根据key进行分组.相同key是一组 value是对应key的元组
   val grouped: Map[String, List[(String, Int)]] = tuples.groupBy(x => x._1)
  //开始统计单词的个数,此时需要生一个新的Map key 对应的单词 value是 单词的个数

    val sum: Map[String, Int] = grouped.map(x => (x._1,x._2.size))

    println(sum)
   //可以直计算value的值
   val sumed: Map[String, Int] = grouped.mapValues(x =>x.size)
  //println(sumed)
  //排序
    val list: List[(String, Int)] = sum.toList
  // 根据传入的数据进行排序 默认是升序 reverse翻转降序
   val sortedList: List[(String, Int)] = list.sortBy(_._2).reverse
    println(sortedList)

    //简化版本
   val list1 = lines.flatMap(_.split(" ")).map((_,1)).groupBy(_._1).mapValues(_.size).toList.sortBy(_._2).reverse
   println("list1 =" + list1 )
}

```

```scala
/**
  * 在scala中 类并不用声明为public,一个文件中那个可以有多类
  * 如果类没有定义构造方法,类会默认给一个空参的构造方法(无参构造方法)
  * 描述类 ,不能写 main
  */
class Student {
    //var修饰的变量就是可读可写,相当于有get和set方法
    //在类中定义的变量并不会直接给初始值,此时就是可使用 _ 表示一个占位符
     //若使用 _ 这种形式 必须显式的声明出当前变量的数据类型
     var name:String = _
      //val修饰变量 是常量 使用了默认占位 说明这个常量就不能在赋值了 没有任何意义
    // val age:Int = _
      val  age = 19
}
//执行类
object Test{
  val  TestName:String = "老王"
  def main(args: Array[String]): Unit = {
     //创建对象 是无参构造方法 此时可以添加() 也可以不添加()
     val student = new Student()
    // val student1 = new Student
     student.name = "小王"
    // student.age = 19 val不能再修改了
    println(student.name +" "+student.age)
    println(Test.TestName)


  }
}

/**
  * 构造方法
  * scala中的构造方法需要类名的后面,主构造方法,一个类中只有有一个主构造方法
  * 但是一个类中可以有多个辅助构造方法
  * 给构造方法参数 修饰 使用 val 和 var 修饰
  * val修饰就像相当于是一个常量提供了get方法
  * var修饰就是相当于是一个变量提供了get和set方法
  * 在构造方法中不是用任何修饰符修饰变量,那么只能在当前中进行访问,伴生对象也不能访问
  * 总结:
  * 在类中书写构造方法需要写在类名的后面,这个构造方法称为主构造方法
  * 并且构造方法中可以定义属性(属性:数据类型)若不使用val或var修饰
  * 当前属性只能在类的内部使用并且默认是val修饰是常量只能访问不能赋值
  * 若使用val 或 var修饰就相当于可以在外部进行访问并且遵守val和var语法
  *  val 就是只有get方法  var 就是既有get也有set
  */
class Student1(val name:String, var age:Int, faceValue:Int = 95) {
  //在类中定义属性
  var gender:String = _

  //辅助构造方法  def this
  //在辅助构造方法中必须在第一行调用主构造方法
  def this(name:String,age:Int,faceValue:Int,gender:String){
      this(name,age,faceValue)
      this.gender = gender
  }

}
object  Test1{
  def main(args: Array[String]): Unit = {
     //若在声明的构造方法中不是var 或 val 修饰这个成员变量,创建的对象是无法访问
      val stu = new Student1("静静",18,100)

      val stu2 = new Student1("我想",20,5,"小孩")

  }
}

```

```scala
/**
  * class的类称为描述类 object类称为执行类(单利类)
  * class类不能写main方法一般是描述居多
  * object类是可以写main方法,在Scala中是没有静态的概念
  * scala创建一个单利对象作为程序的入口提供点,就是object的类
  * object类不创建单例对象,代码也会变异成,但不会有任何输出,单例对象中声明的方法可以全局访问
  * 当object类作为伴生对象的时候可以扩展原有的类
  *
  * java中单例
  * 私有化构造方法
  * 提供一个私有化的属性
  * 饿汉: 在提供属性的同时并初始化
  * 懒汉: 只声明不创建
  * 提供一个当前类类型的静态方法,获取当前属性
  * 饿汉: 直接返回创建好的属性即可
  * 懒汉: 先判断属性时候为空,若为空就创建 ,否则返回(注意线程安全)
  *
  * 简化版本
  * 私有化构造方法
  * 创建一个公有静态不可变的当前类类型的并赋值即可
  *
  * 枚举版本
  * 声明属性(创建枚举常量即可)
  *
  *
  * scala中使用object修饰的类就是单例类,创建出来的对象就是单例对象
  * object中的属性和方法看做类似于java中的静态方法和静态成员,可以用类调用
  *
  */
object Single{
  private val instance = Single
  def getInstance= instance

}
object SingleDemo {
   def main(args: Array[String]): Unit = {
     val sd1 = Single.getInstance
     val sd2 = Single.getInstance
     println(sd1 == sd2)
   }
}

```

```scala
/*
伴生类和伴生对象
在Scala中当单例类和某个类共享一个名称的时候,单例类被称为这个类的伴生类,创建的对象就是伴生对象
必须在同一个源文件里定义类和它的伴生类,伴生类和类之间可以互相访其私有属性
伴生类类看做是普通类的一个扩展
Apply和unApple
这两个方法时应用于伴生类中
apply通常被称为注入方法,在类的伴生对象中做一些初始化操作
apply方法的参数列表不需要和构造方法的参数列表统一
unapply通常被称为提取方法,使用unapply方法可以提取固定数量的对象或值
ps:unapply这个方法的返回值是一个Option类型
 有两个子类 Some 有值 和 None 没有值
 apply方法和unapply反方都是被隐式调用的
 */
class ApplyDemo(var name:String,var age:Int){

}
object ApplyDemo{
   //注入方法
   def apply(name:String,age:Int):ApplyDemo ={
     new ApplyDemo(name,age)
   }
   //提取方法
    def unapply(applyDemo:ApplyDemo):Option[(String,Int)]={
        if(applyDemo == null){
          None
        }else{
          Some(applyDemo.name,applyDemo.age)
        }
    }
}
object  ApplyTest{
  def main(args: Array[String]): Unit = {
    //apply方法被调用
      val applyDemo = ApplyDemo("xiaohong",18)
    //scala是没有switch-case的 但是提供了一种匹配模式 match
     applyDemo match{
         //调用unapply
       case ApplyDemo(name, age)=> println(name+" "+age)
       case _ => println("什么都没有")

     }
  }
}

class AssociatedObject {
    var id = 0
    private  val  name = "张三"
    def printlns() ={
       println(name+" "+id+" "+AssociatedObject.gender)
    }
}
object AssociatedObject{  //这个类就是上面这个类的伴生类
    private val gender = "boy"

  def main(args: Array[String]): Unit = {
    //创建伴生对象
    val ao = new AssociatedObject
    //val ao1 = AssociatedObject
    println(ao.id)
    println(ao.name)
    ao.printlns()

  }
}
```

```scala
/*
抽象类
在Scala中 使用abstract修饰的类就是抽象类,在抽象类中可以定属性,没有实现的方法
 */
abstract class Animal {
   val age = 5 //声明一个有值的字段
   val name:String //声明一个没有值的字段
   val gender =  "kkk"
   def eat():String={ //定义实现方法
     "吃饭"
   }
   def run():String //定义实现没有实现的方法
}
/**
  *这是scala特有的修饰 直接翻译过来是特质的意思
  * 直接看做是java中接口即可
  * scala中说:trait要比java的接口强,除了可以定义属性外 还可以定义实现的方法
  * ps:强☞1.7之前的JDK  1.7之后提供了新的接口没有任何区别
  */
trait Fly{
  val height:Int = 10000 //声明并赋值
  val speed:Int //没有值的
   def fly():String={ //定义实现方法
     "I can fly"
   }
  def sleep():String //睡觉
}
/**
  * 无论是抽象类 还是 特质 若使用类来继承或实现 都是用同一个关键 extends
  *  继承一个类并实现多个特质
  *  extends 类 with 特质
  *  若有多个extends 类 with 特质 with 特质 ......
  *  父类已经实现功能 子类若实现需要使用 override关键字重写
  *  父类没有实现的方法子类必须实现
  */
class Person extends Animal with  Fly {
   //atl+enter 提示
  // ctrl+atl+鼠标左键 看实现了
  //快速查询某个类 2次shift

  //重写已经提供过的属性或方法
  override val age: Int = 10

  override val name: String = "小强"

  override def run(): String = "我想飞"

  override val speed: Int = 1000

  override def sleep(): String = "睡着了"
  // variable gender cannot override a mutable variable
  //子类只能重写父类val修饰的属性
  override val gender = "11111"

}
//Trait可以继承class
class MyUtil {
    def printMSG(msg:String)=println(msg)
}
trait MyTrait extends MyUtil{

}
//myUtil类是所有继承(实现)MyTrait父类
class Demos extends  MyTrait{
}
```

```scala
//并行化集合 par
object ParDemo {
  def main(args: Array[String]): Unit = {
      val arr = Array(1,2,3,4,5,6,7,8,9,10)
    //求和:调用par方法此时会有多个线程同时进行聚合计算
    //每一个线程只算一部分,然后最终在聚合 (1+2+3+4)+(5+6+7+8)+(9+10)
    // sum 是一个聚合函数(求和)
     val sum1: Int = arr.par.sum
     println(sum1)
      val sum11:Int = arr.sum
    println(sum11)
    //通过reduce方法进行参数的合并
    //第一个x是数组中一个元素的值 第二个y是数组第二个元素的值,然后相加
    //得到的结果会被x再次获取,y会回去数组下一个值然后相加到结束
     val sum2: Int = arr.par.reduce((x,y)=>x+y)  // 简化arr.par.reduce(_+_)
     println(sum2)
    val sum22:Int = arr.reduce(_+_)
    println(sum22)

    //启动单线程进行相减
     val sub : Int = arr.reduce(_-_)
      println(sub)
    //并行化好处理结果
    //统一时间之内开启多个线程进行计算,最终聚合
     val sub1 :Int = arr.par.reduce(_-_)
    println(sub1)
    //reduceLeft方法带调用par的时候,依然还是单线程循序计算 从左向右
    val sumd: Int = arr.reduceLeft(_+_)
    val sumd1:Int = arr.par.reduceLeft(_+_)
    println(sumd)
    println(sumd1)
    val sub2 : Int = arr.par.reduceLeft(_-_)
    println(sub2)
    //reduceRight//从右向左

    //合并: join
    //交集.并集,差集
    val l1 = List(5,6,4,7)
    val l2 = List(1,2,3,4)
    //求并集,给定的A,B两个集合 把他们所有的元素合并在一次组成的集合就叫做并集
    val res :List[Int] =  l1 union l2
    println(res)
    //求交集:给定A,B两个集合,由所有属于集合A且属于集合B的元素所组成的集合叫做交集
     println(l1 intersect l2)
    //求差集:给定A,B两个集合,由所属于集合A且不属于集合B的元素构成的集合,叫做差集
    println(l1 diff l2)

    //下面这个方法使用高阶函数(柯里化)
    //折叠 类似于reduce和sum 都是聚合函数
    /*
     第一个小括号中参数是初始值 第二个小括号中参数是计算规则
     这个初始值是0的前提下无论如何计算都会是一样
     这个初始值在计算的时候会被运算相加
     第一个_第一次获取的是初始值 第二个_先获取的是集合中值 然后计算
     */
    val sum5: Int = arr.fold(10)(_+_)
    println(sum5)
    //若fold开启并行,需要注意每个线程计算的时候都回去获取这个初始值
    val sum6:Int = arr.par.fold(10)(_+_)
    println(sum6)
    //也是单线程计算计算式开启并行
    println(arr.foldLeft(10)(_+_))
    println(arr.par.foldLeft(10)(_+_))
    //有一个兄弟arr.foldRight效果是一样的
    //需求将下面集合中的数据求和
    val arr1 = List(List(1,2,3),List(4,5,6),List(111))
    val i1: Int = arr1.flatten.reduce(_+_)
    val i2: Int = arr1.flatten.fold(0)(_+_)
    // arr1.aggregate(0)(_+_.reduce(_+_),_+_)

  }
}
```

# day4

## private 私有化    

```
private加在主构造方法上, 外部不能访问这个构造方法了
class 类名 private(参数){ 提供一个辅助构造器 def this(){}}
private添加在成员变量,可以其内部访问伴生类也可以访问
private[this]添加成员变量, 只能在类的内部使用,伴生类也无法访问
private[包名] class 类名 当前只能在这个包下可以访问
```

## final关键字和type关键字    

```
被final修饰类不能继承,方法不能被重写
type关键字可以用来声明类型(得到一个类型,类型的别名)
type s = 数据类型
type s = String //S就能当做String 来使用
val name : String = "太阳"
val name : s = "月亮
```

## 类型判断和模式匹配    

```scala
Scala中 一切类的基类(超级父类)Any
AnyRef:是Any的子类,是所有引用类型父类,除了值类型,所有的类型都继承于它
AnyVal:是Any的子类,是所有值类型的父类,它是描述值的,但是不代表一个对象
AnyVal的子类 9个
Byte,Short,Int,Long,Float,Double,Char 七种值类型
Unit,Boolean 是非值类型
引用类型的一些操作
判断当前对象是否是当前类型的实例
Java obj instanceof Object Scala obj.isInstanceOf[object]
将当前对象进行强制类型转换
Java (Object)obj Scala obj.asInstanceOf[object]
获取当前类型的Class对象
Java obj.Class Scala classOf[obj]
模式匹配
Scala是没有Java中switch-case语法,但是相对应的.Scala提供另一个更加强加的匹配模式语法
match-case,用来替代switch-case
Scala中match-case和Java红switch-case最大的不同点在于:
Java中swithc-case仅能匹配:数值不能是long不能是浮点类型,字符串,枚举
scala中match-case可以匹配各情况,比如:变量的类型,集合的元素, 有值或无值
语法:
匹配的变量 match{
case 值 => 代码
}
case 可有多个,若使用的是下划线,下划线以上的有case都没有满足,处理剩余情况
match-case不需要break,只要有一个case分支满足了剩余的就不会判断了
ps:switch-case --> default其实就是这个下划线 最后一种情况
案例:

import scala.util.Random
object MatchCaseDemo {
  def main(args: Array[String]): Unit = {
    val arr1 = Array("YuiHatano","AoiSola","YoshizawaAkiho")
     arr1(Random.nextInt(arr1.length))match {
       case "YoshizawaAkiho" => println("吉泽老师......")
       case "YuiHatano"  => println("波多老师")
       case _ => println("我都不知道你们在说什么,不懂......")
     }
     println("---------------------------------华丽的分割线------------------------------")
    val arr2:Array[Any] = Array("hello123",1,2.0,2L,MatchCaseDemo)
    arr2(Random.nextInt(arr2.length))match {
      case x:Int => println("Int类型"+x)
      case y:Double => println("Double类型"+y)
      case z:Long => println("Long类型"+z)
      case str:String => println("String类型"+str)
      case MatchCaseDemo => println("当前类类型")
      case _=> println("没有这个类型")
    }
    println("---------------------------------华丽的分割线------------------------------")
    val arr3 = Array(1,1,7,0,2,3)
     arr3 match{
         //数组中的元素要是能匹配成功需要是 0和2这两个元素开头
       case Array(0,2,x,y) => println("x="+x+" " + "y="+y)
       case Array(1,1,7,_*) => println("匹配成功")//_*任意多个
       case _ =>println("最后一种情况")
     }
    val list = List(1,2,3)
    println(list.head)//头部数据
    println(list.tail)//获取尾部数据
    println("---------------------------------华丽的分割线------------------------------")
    // :: 拼接头部到集合中
    list match{
      case 0 :: Nil => println("只有0")
      case x :: y :: Nil => println("有x和y")
      case 1 :: a => println(a)
      case _ => println("没有")
    }
    println("---------------------------------华丽的分割线------------------------------")
    val tup = (1,3,7)
    tup match {
      case (3,x,y) => println(x+y)
      case (x,y,z) =>  println(x+" "+y+" "+z)
      case (_,w,5) => println(w)
      case _ => println("没有")

    }
  }
}

```

## 样例类    

```scala
是模式匹配中的特殊类,会配合case class和 case objetc来使用
样例类,使用case关键字修饰的类,可以支持模式匹配,样例类默认实现了序列化接口可以把样例类看做是java中的枚举加
import scala.util.Random
/*
样例类基本使用
*/
object CaseYangli {
def main(args: Array[String]): Unit = {
val arr = Array(CheckTimeOutTask,SubmitTask("00001","task_00001"),HeatBeat(150000))
//进行匹配若匹配成功就输出对应的值
arr(Random.nextInt(arr.length)) match{
case CheckTimeOutTask => println("CheckTimeOutTask")
case SubmitTask(id,task) => println("SubmitTask")
case HeatBeat(time) => println("HeatBeat")
}
}
} /
* 样例
类有两种声明方式 但是都是以case开头
若case 后面是object创建类是不支持封装数据,但是可以支持匹配
ps:实际案例需要在实现akka的案例中来演示
*/
case object CheckTimeOutTask
case class SubmitTask(id:String,taskName:String)
case class HeatBeat(time:Long)
```

## Option类型    

```scala
用阿里表示可能有的值和没有值的情况,默认是有两个子类 Some有值None没有值
Some类型可以一些值
使用map存储值通过get方法获取数据 会出现一个异常问题,key不存在是取值失败跑出异常
getOrElse就会得到当前的值,通过转换拿到具体的值
getOrElse主要就是一个防范措施,获取得到值就是Some,没有就是None
例子:
val seq:Option[(String,Int,Boolean)] = Some("xiaoming",18,true)
//会给一个参数 这个参数是一个默认值,也即是说若取值失败就返回 null ,取值成功就是值
val value:(String,Int,Boolean) = seq.getOrElse(null)
```

## 偏函数    

```scala
被包在花括号内部但是没有match关键字的一组case语句就属于偏函数
/**
* 偏函数
*/
object PartialFunctionDemo {
// 普通版本的匹配模式
def m1(mun:String):Int = mun match {
case "one" => 1
case "two" => 2
case _ => -1
} /
/可以将上面的代码进行简化
//偏函数的名字是固定的不能修改PartialFunction
//第一个是参数的数据类型,第二个是返回值的数据类型
def m2:PartialFunction[String,Int]={
//若case有多行语句建议使用{ }括起来分行写
case "one" =>{
println("case one")
1
} c
ase "two" =>{
println("case one")
2
} c
ase _ => -1
} d
ef main(args: Array[String]): Unit = {
println(m1("two"))
println(m2("one"))
}
}
```

## 获取线程池中的返回值Future    

```scala
import java.util.concurrent.*;
//获取线程池中的数据
public class FutureDemo {
public static void main(String[] args) throws ExecutionException, InterruptedException {
//线程池4种创建方式
// 创建一个定长的线程池,可以控制线程的最大并发数,
//超出的线程会在队列中的等待
//定长线程池的大小最好根据系统资源进行设置
//Runtime 运行时类
// int i = Runtime.getRuntime().availableProcessors();
// System.out.println(i);
// //创建定长的线程池
// ExecutorService threadPool = Executors.newFixedThreadPool(10);
//创建一个可缓冲的线程池,如果线程池的长度超过处理需要,可以灵活回收空闲线程,若无回收,则创建信息的
线程
//线程池基本上是无限大,当执行第二个任务的时候,若第一个任务已经完成,会复用执行第一个任务的线程,而
不会每次都新创建线程
ExecutorService threadPool = Executors.newCachedThreadPool();
//关闭线程池
//threadPool.shutdown();
//Future相当于一个容器,可以封装返回值
//当计算过没有完成的时候,此时会线程阻塞,Future在线程阻塞时为空
//submit方法时有返回值,这个返回值就Future类型
Future<String> future = threadPool.submit(new Callable<String>() {
@Override
public String call() throws Exception {
System.out.println("Thread name:" + Thread.currentThread().getName() + " " +
"Thread id" + Thread.currentThread().getId());
System.out.println("正在读取数据....");
Thread.sleep(1000);
System.out.println("读取数据完毕");
return "success";
}
});
//因为main的有优先级高,所以为了的能等到资源
Thread.sleep(2000);
//获取数据先判断是否有数据 true即使有数据 false就是没有数据
if(future.isDone()){
System.out.println(future.get());
}else{
System.out.println("None");
}
}
}
```

## Actor    

```scala
Actor是可以实现并行编程,它是基于事件模型的并发机制,运行消息的发送和接收来实现多线程,所以用Actor可以实现
多线程应用开发
Actor模型详解
在Scala2.10之前直接使用Actor.在2.11版本中添加Akka 作为默认Actor
Scala中Actor的特点:
Actor不共享数据,没有锁的概念,Actor之间通过Message来进行通信
Scala中Actor的执行顺序
调用start()方法来启动Actor,执行Actor的act()方法,Actor发送消息
Actor的发送消息3种方式
! 发送异步消息,没有返回值
!? 发送同步消息,有返回值,线程会等待
!! 发送异步消息,有返回值,返回值是用Future[Any]接收的
什么是同步什么是异步?
异步调用是通过单独的线程执行,原始线程启动启动异步调用,异步调用使用一个线程执请求,而与此同时原始的线程继续处理
同步调用则在继续之前必须等待相应或返回值,会开启新线程,但是需要阻塞主线程的到反馈的结果或返回值,不允许得到无
返回值或无相应的效果,不然就会一直阻塞
import scala.actors.Actor

//简单使用Actor
object ActorDemo {
  def main(args: Array[String]): Unit = {
    //启动Actor
    MyActor.start()
  }
}
object MyActor extends Actor{
  //具体实现的内容
  override def act(): Unit = {
    for(i <- 1 to 10){
      println(i)
      Thread.sleep(1000)
    }
  }
}
import scala.actors.{Actor, Future}


object ActorTest{
  def main(args: Array[String]): Unit = {
      //得到启动过后的Actor
      val actor = ActorDemo2.start()
     //1. 步发送异没有返回值 !
    //使用actor对象并发送数据
//    actor ! AsyncMsg(1,"hello hello")
//    //因为是异步执行所有不会等到上面执行完成后才会执行下面这句话
//    println("异步消息没有返回值")

    //2.同步发送信息 会有返回值 线程会等待 !?
//    val msgs: Any = actor !? SyncMsg(2,"Bye Bye")
//     println("同步消息发送完成")
//    println(msgs)

    //3.异步发送信息,并且有返回值,返回值是Future[Any]
      val reply: Future[Any] = actor !! AsyncMsg(3,"good good")
      //判断是否有值,ture 就证明有值 false 就是没有值
       //因为是异步访问,所以这里需要做一个延迟操作 相当于等待异步线程的返回
        Thread.sleep(1000)
    if(reply.isSet){
      //获取消息
     val value: Any = reply.apply()
      println(value)
    }else{
      println("Nothing...")
    }

  }
}


//处理逻辑,使用Actor进行消息发送
object ActorDemo2 extends Actor{
  override def act(): Unit = {
     while(true){
          //要接收数据 偏函数
       receive{
         //异步发送过来的 id和信息
         case AsyncMsg(id,msg) =>{
             println(id + " "+ msg)
           //发送消息 接收到信息之后会发一条
           /*
             3种:
             !  发送异步信息, 没有返回值
             !? 发送同步想你想, 有返回值 并线程会等待
             !! 发送异步信息,有返回值, 返回值类是Future[Any]
            */
           sender ! ReplyMsg(3,"success")
         }
         case SyncMsg(id,msg) => {
           println(id + " "+ msg)
           //人工睡眠一会
           //Thread.sleep(3000)
           //发送消息 接收到信息之后会发一条
           sender ! ReplyMsg(4,"success")
         }

       }

     }

  }
}
//异步
case class  AsyncMsg(id:Int,msg:String)
//同步
case class SyncMsg(id:Int,msg:String)
//回传信息
case class ReplyMsg(id:Int,msg:String)

```

## 高阶函数    

```scala
作为值的函数,匿名函数,闭包,柯里化,隐式转换函数等等
高阶函数的实现使用是AOP思想(面向切片):可以理解为是面向对象的一种补充
面向切片主要作用是把一些跟核心业务逻辑模块无关的功能抽取出来
在通过"动态织入"方式传入业务逻辑模块中
AOP通俗解释
高阶函数
将其他函数作为参数或结果是一个函数
这里的f是函数名 相当于一个函数 v是这个f函数中的一个参数
def function(f:Int=>String,v:Int) = f(v)
1.作为值的函数:
函数的定义,调用和转换
val func1 = (x:Int) => x * 2
简化这个函数
val func2:Int => Int = x => x * 2
调用
val arr = Array(1,2,3,4)
arr.map(func1)
func1(1)
匿名函数只会调用一次
arr.map(x => x * 2)
转换:
def show():Unit={println(1)}
val func3 = show _
ps:在实际使用中隐式的转换方法到函数
def method(x:Int):Int ={ x * 2}
val arr1 = Array(1,2,3,4)
arr1.map(method) //隐式的转换为函数
2.作为闭包
闭包是一个函数,返回值依赖于声明在函数外部的一个或多个变量
闭包通常来讲可以简单的而你我是可以访问一个函数里面局部变量的另外一个函数
//闭包
object BibaoDemo {
  def main(args: Array[String]): Unit = {
     //局部变量
    var i = 3
    //定义一个函数 这个函数访问了外部的局部变量 这个 函数就是闭包
    val sum = (j:Int) => i+j
    println("sum="+sum(3))
  }
}

//闭包
object BibaoDemo2 {

  def showA(msg:String) = (name: String) => println(name + msg)

  def main(args: Array[String]): Unit = {
    val f: String => Unit = showA("我是闭包")
    f("张三")

  }
}

```

## 柯里化    

```scala
是把接收多个参数的函数转换成接收一个单一参数的函数
指的是就原来接收连个参数的函数变成新的接收一个参数的函数过程
新的函数返回一个以原有第二个参数为参数的函数
//柯里化
object CurryDemo {
  def main(args: Array[String]): Unit = {
    //声明柯里化的方式有两种
    //第一种:以implicit关键字带有单哥参数的函数
    //定一个方法方法可以求和
    def curry1(x:Int,y:Int) = x + y
    //调用方法
    println(curry1(1,2))
    //此时我们将当前这个方法进行一个变化,结果是不变,参数改变,这种方式就是柯里化
    def curry2(x:Int)(y:Int) = x + y
    //调用方法:
    println(curry2(1)(2))

    //继续变形可以给其中的一个参数进行赋值
    def curry3(x:Int)(y:Int = 2)=x + y
    println(curry3(1)())
    //其中参数implicit关键字来修饰
    //implicit 隐式转换,会自动帮组计算第二个参数
    //不使用第二个参数可以自己传入,若不传自动计算还能完成类型转换
    def curry4(x:Int)(implicit y:Int = 2) = x+y
    println(curry4(1))

    //需求:将数组中的value相加
    val arr = Array(("tom",1),("jerry",2),("kitty",4))
    //reduce不能使用的原因是因为这里参数类型不配
    /*
     这个fodLeft是一个柯里化参数的典型应用
     第一个参数是一个初始值 foldLeft是可以进行并行化计算par,单线程说初始值记性一次加载
     第一次计算的时候 x的值就是 0  y是数组中元素  0+第一个元组中的值
     得到结果后因为这个方法的返回值是初始值类型所 下一次x 获取的即是上一次计算的值
     又因为foldLeft也会遍历这个数组中的元素(即第二个参数)所以会计算这里的所有值
     */
    //val i: Int = arr.foldLeft(0)((x, y)=>x+y._2)
    //简化
    val i:Int = arr.foldLeft(0)(_+_._2)
    println(i)

  }
}
object CurryDemo2 {
  def main(args: Array[String]): Unit = {
    //第二种声明方式即新的函数返回的是以原有第二个参数为函数的
    def curry1(x:Int) = (y:Int) => x+y
    //上面的这种实现其实就是下面柯里化的一个翻版
    def curry2(x:Int)(y:Int) =  x + y
  }
}

```

# day5

## 隐式转化 

```scala
隐式参数和隐式转换函数
作用:利用隐式转换可以实现优雅的类库
ps:隐式转换确实可以增强方法,丰富类库
隐式转换函数:
以implicit关键声明的带有单个参数的函数
系统中提供了一个隐式类Predef类它的父类LowPriorityImplicits,系统提供了78种隐式转换
Int类 整数类型 to 整数类型 ,在Int类中是没有,这个方法是RichInt类提供
Java中有23种设计模式: 单例 ,工厂,模板,装饰者,代理
Scala中的隐式转换其实就是 装饰+门面设计模式
ps:门面叫法的不同 --> 外观设计模式
门面:是指提供一个统一的接口去访问多个子系统的多个不同的接口[看图]
问题: 继承,代理和装饰者的概念和不同
继承:通过继承父类可以的到基础信息,可以扩展父类并且对父类中的方法提供加强
代理:远程代理一个实例,可以对实例方法进行增强
装饰者:使用IO流的时候,Buffer带缓冲区的,装饰者模式
import scala.io.Source
//对file类进行增强,通过路径直接获取数据
class RichFile(val file:String) {
//提供一个自己的方法,直接读取数据
def read():String = {
//mkString是scala中的toString方法
Source.fromFile(file).mkString
}
} /
/自己实现的时候最好使用object,类名随便
object MyPredef {
//隐式转换函数的的连接
//implicit def FileToRichFile(file:String) = new RichFile(file)
implicit val FileToRichFile = (file:String) => new RichFile(file)
}
```

## 泛型

```scala
cala中的泛型可以应用到类,方法和函数中,泛型占位符本身没有任何意思,只有实际传入值的时候才会有具体提的意义
存在
Scala中书写方式不同,要使用泛型需要使用[占位符]即可
Scala中泛型给了一些名词: 上界,下界,视界,上下文界
以下B和A都是占位符没有实际意义
[B <: A]:上界或上限 : 表达了泛型B必须是A类型或是A的子类
[B >: A]:下界或下限 : 表达了泛型B必须是A类型或是A的父类
[B <% A]:视界: 可以进行隐式转换,把当前B类型转换为A类型
[B : A]: 上下文界 ,进行隐式转换,把当前B类型转为A类型
[-A,+B]
-A:(逆变)可以传入其父类或本身---> 参数类型
+B:(协变)可以传入其子类或本身---> 作为返回值类型
基本泛型使用:
class Teacher(val name:String,val faceValue:Int)extends Comparable[Teacher]{
override def compareTo(that: Teacher): Int = {
//升序 当前对象-传入对象
//降序 传入对象-当前对象
that.faceValue - this.faceValue
}
} 
object Teacher {
def main(args: Array[String]): Unit = {
val t1 = new Teacher("小苍老师",90)
val t2 = new Teacher("小米老师",faceValue = 91)
val arr = Array(t1,t2)
val sorted: Array[Teacher] = arr.sorted
println(sorted(0).name)
}
    } 上
界实现
class Girl(val name:String,val faceValue:Int,val age:Int)extends Comparable[Girl]{
override def compareTo(that: Girl): Int = {
//升序 当前对象-传入对象
//降序 传入对象-当前对象
if(this.faceValue == that.faceValue){
that.age - this.age
}else{
this.faceValue - that.faceValue
}
}
} /
/提供一个选择的类
class UpperBoundsDemo[T <: Comparable[T]] {
def XuanZhe(first:T,Second:T): T ={
if(first.compareTo(Second) > 0) first else Second
}
} /
/选择实现类
object UpGirl {
def main(args: Array[String]): Unit = {
val xz = new UpperBoundsDemo[Girl]
val g1 = new Girl("小红",90,35)
val g2 = new Girl("小绿",80,30)
val girl: Girl = xz.XuanZhe(g1,g2)
println(girl.name)
}
} p
s:下界和上界没什么区别都是一样就含义不同
视界
class Boy(val name:String,val faceVlaue:Int,val age:Int) {
} /
/视界提供一种隐式转换的方式
//作为函数的参数存在(隐式参数 ) 作为隐式函数存在
/*
Ordered是scala排序 --> 实现的是Java中Comparable接口
Ordering是Scala排序 --> 实现的是Java中Comparator接口
视界可以看做是对上界或下界的一个增强
*/
class ViewBoundsDemo[ T <% Ordered[T]]{
def Xuanzhe(first:T,second:T): T ={
//方法体中代码只有一个行,if有自带返回值,此时可以不写return
if (first > second) first else second
//这里还有执行,上面的语句返回就需要使用return
//if(first < second) second else first
}
}object ViewBoundsDemo{
def main(args: Array[String]): Unit = {
//需要导入隐式转换
import MyPredefOrder.BoySelect
val vbd = new ViewBoundsDemo[Boy]
val b1 = new Boy("张三",50,49)
val b2 = new Boy("李四",55,50)
val boy: Boy = vbd.Xuanzhe(b1,b2)
println(boy.name)
Array().sorted
}
} /
/这个类必须是单利类object
object MyPredefOrder {
//在完成Ordered中对数据的排序
implicit val BoySelect = (boy:Boy) => new Ordered[Boy]{
override def compare(that: Boy): Int = {
if(boy.faceVlaue != that.faceVlaue){
boy.faceVlaue - that.faceVlaue
}else{
that.age - boy.age
}
}
}
} 上
下文界
//上下文界 和 视界类似 有一个隐式转换(隐式参数/隐式函数)
// [B : A] Ordered / Ordering 排序 classTag:相当动态传入类型,你使用什么类型传就是什么类型
//例子[B : Ordering] 或 [B : classTag]
class ContextBoundsDemo[T : Ordering] {
def Xuanzhe(first:T,second:T): T ={
val ord : Ordering[T] = implicitly[Ordering[T]]
if (ord.gt(first,second)) first else second
}
} o
bject ContextBoundsDemo{
def main(args: Array[String]): Unit = {
//需要导入隐式转换
import MyPredefOrdering.OrderingBoy
val vbd = new ViewBoundsDemo[Boy]
val b1 = new Boy("张三",50,49)
val b2 = new Boy("李四",55,50)
val boy: Boy = vbd.Xuanzhe(b1,b2)
println(boy.name)
}
} o
bject MyPredefOrdering {
//这个是上下文界限中需要使用的实现方式
implicit object OrderingBoy extends Ordering[Boy]{
override def compare(x: Boy, y: Boy): Int = {
if (x.faceVlaue == y.faceVlaue){Akka
RPC通信模型
y.age -x.age
}else{
x.faceVlaue - y.faceVlaue
}
}
}
}
```

## Akka 

```scala
package AkkaDemoimport akka.actor.{Actor, ActorRef, ActorSystem, Props}
import com.typesafe.config.{Config, ConfigFactory}
//创建Master类
class Master extends Actor{
//方法只会执行一次,构造方法之后 receive之前
override def preStart(): Unit ={
println("方法执行了")
} /
/必须实现 用来接收数据和发送数据使用 在perStart之后执行 会一直执行
override def receive: Receive = {
case "start" => println("接收到自己发送的信息")
case "stop" => println("停止")
case "connect" => {
println("connect")
sender ! "reply"
}
}
} o
bject Master{
def main(args: Array[String]): Unit = {
//IP地址和端口号
val host = "127.0.0.1"
val port = "6666"
//当前字符串其实是配置信息
//s是引用字符串中变量的值 """ 使用封装kv形式的配置信息
// | 用来分割对应kv信息使用
val configStr =
s"""
|akka.actor.provider = "akka.remote.RemoteActorRefProvider"
|akka.remote.netty.tcp.hostname = "$host"
|akka.remote.netty.tcp.port = "$port"
""".stripMargin
//配置创建Actor需要的配置信息
val config: Config = ConfigFactory.parseString(configStr)
//创建ActorSystem
val masterSystem: ActorSystem = ActorSystem("MasterSystem",config)
//用masterSystem创建Actor对象
val master: ActorRef = masterSystem.actorOf(Props[Master],"Master")
//为了检查当前master是否正常
master ! "start"
}
} p
ackage AkkaDemo
import akka.actor.{Actor, ActorRef, ActorSelection, ActorSystem, Props}
import com.typesafe.config.{Config, ConfigFactory}
//worker端
class Worker extends Actor{
//worker请求Master使用AKKA实现SparkRPC
override def preStart(): Unit = {
//通过Master的URL获取Master的Actor对象
//user语法规定请求那个角色
val master: ActorSelection =
context.actorSelection("akka.tcp://MasterSystem@127.0.0.1:6666/user/Master")
master ! "connect"
} o
verride def receive: Receive = {
case "self" => println("接收到自己的信息")
case "reply" => println("接收到Master端发过来的信息")
}
} o
bject Worker{
def main(args: Array[String]): Unit = {
//IP地址和端口号
val host = "127.0.0.1"
val port = "8888"
//当前字符串其实是配置信息
//s是引用字符串中变量的值 """ 使用封装kv形式的配置信息
// | 用来分割对应kv信息使用
val configStr =
s"""
|akka.actor.provider = "akka.remote.RemoteActorRefProvider"
|akka.remote.netty.tcp.hostname = "$host"
|akka.remote.netty.tcp.port = "$port"
""".stripMargin
//配置创建Actor需要的配置信息
val config: Config = ConfigFactory.parseString(configStr)
//创建ActorSystem
val workerSystem: ActorSystem = ActorSystem("WorkerSystem",config)
//用masterSystem创建Actor对象
val worker: ActorRef = workerSystem.actorOf(Props[Worker],"Worker")
//为了检查当前worker是否正常
worker ! "self"
}
}
```

## 使用AKKA实现SparkRPC 

```scala
1.首先workerInfo类 主要分装着work向master注册的信息 基本信息之外还提供心跳
2.创建一个特质存储的都是用来进行数据传递的样例类
3.创建Master类来进行逻辑处理
4.创建worker类来进行逻辑处理
代码看SparkRPC
package RPCDemo
//存储work注册的信息和心跳信息
class WorkerInfo(val id: String, val host: String,
                 val port: Int, val memory: Int, val cores: Int) {

  // 记录最后一次心跳时间
  var lastHeartbeatTime: Long = _
}


package RPCDemo
//继承序列化接口是为了传输数据
trait RemoteMsg extends Serializable{

}
//进行匹配的
// Master 向self发送的信息
case object CheckTimeOutWorker

// Worker 向 Master发送的信息 从网络发送所以哟啊序列化 所以要继承RemoteMsg
case class RegisterWorker(id: String, host: String,
                          port: Int, memory: Int, cores: Int) extends RemoteMsg

// Master向 Worker信息(注册成功之后)
case class RegisteredWorker(masterUrl: String) extends RemoteMsg

// Worker 向 Master信息 (注册成功后的心跳)
case class Heartbeat(workerId: String) extends RemoteMsg

// Worker 向 self 发送信息
case object SendHeartbeat



package RPCDemo

import akka.actor.{Actor, ActorSystem, Props}
import com.typesafe.config.{Config, ConfigFactory}
import scala.collection.mutable
import scala.concurrent.duration._
//需要初始化Host和Port地址  这些变量用到的时候在补全
class Master(val masterHost: String, val masterPort: Int) extends Actor{

    // 用来存储Worker的注册信息 vlaue需要多个值所以需要在封装一个类型并创建这个类
    val idToWorker = new mutable.HashMap[String, WorkerInfo]()

    // 用来存储Worker的信息
    val workers = new mutable.HashSet[WorkerInfo]()

  // Worker的超时时间间隔
  val checkInterval: Long = 15000

  // 生命周期方法(先写)
  override def preStart(): Unit = {
    // 启动一个定时器，定时检查超时的Worker
    //import context.dispatcher
    import scala.concurrent.duration._
    //第一个参数是起始时间从0开始 第二个参数超时是间隔时间
    //第三个 receive 给自己发送self  第四个 参数是如何检查worker的逻辑
    //这个时候需要创建一个trait RemoteMsg 并写CheckTimeOutWorker的样例类
    context.system.scheduler.schedule(
      0 millis, checkInterval millis, self, CheckTimeOutWorker)
  }

  // 在preStart之后，反复调用(在写)
  override def receive: Receive = {
    // Worker 向 Master发送的消息 最后两个参数是请求资源用的 内存和核心数
    case RegisterWorker(id, host, port, memory, cores) => {
      //idToWorker不包含(即没有注册信息)
      if (!idToWorker.contains(id)){
        //注册信息
        val workerInfo = new WorkerInfo(id, host, port, memory, cores)
        idToWorker += (id->workerInfo)
        workers += workerInfo

        println("a worker registered")
        //存储成功之后向work发送一个消息
        sender ! RegisteredWorker(s"akka.tcp://${Master.MASTER_SYSTEM}" +
          s"@${masterHost}:${masterPort}/user/${Master.MASTER_ACTOR}")
      }
    }
      //注册成功之后接受心跳
    case Heartbeat(workerId) => {
      // 通过传过来的workerId获取对应WorkerInfo
      val workerInfo = idToWorker(workerId)
      // 获取当前时间
      val currentTime = System.currentTimeMillis()
      // 更新最后一次心跳时间
      workerInfo.lastHeartbeatTime = currentTime
    }
      //检查超时
    case CheckTimeOutWorker => {
      // 获取当前时间
      val currentTime = System.currentTimeMillis()
      //w获取works中的数据 即一个work的信息 当前时间-workInfo的时间 >设置的时间 那就是超时了
      val toRemove: mutable.HashSet[WorkerInfo] =
        workers.filter(w => currentTime - w.lastHeartbeatTime > checkInterval)
      // 将超时的Worker移除
      toRemove.foreach(deadWorker => {
        idToWorker -= deadWorker.id
        workers -= deadWorker
      })

      println(s"num of workers: ${workers.size}")
    }
  }
}
object Master{

  val MASTER_SYSTEM = "MasterSystem"
  val MASTER_ACTOR = "Master"

  def main(args: Array[String]): Unit = {

//    val host = args(0) // localhost
//    val port = args(1).toInt
    val host = "127.0.0.1" // localhost
    val port = "6666".toInt
    val configStr =
      s"""
         |akka.actor.provider = "akka.remote.RemoteActorRefProvider"
         |akka.remote.netty.tcp.hostname = "$host"
         |akka.remote.netty.tcp.port = "$port"
       """.stripMargin

    // 配置创建Actor需要的配置信息
    val config: Config = ConfigFactory.parseString(configStr)

    // 创建ActorSystem
    val actorSystem: ActorSystem = ActorSystem(MASTER_SYSTEM, config)

    // 用actorSystem实例创建Actor
    actorSystem.actorOf(Props(new Master(host, port)), MASTER_ACTOR)
    //线程等待
    actorSystem.awaitTermination()

  }
}

package RPCDemo

import java.util.UUID
import akka.actor.{Actor, ActorSelection, ActorSystem, Props}
import com.typesafe.config.{Config, ConfigFactory}
import scala.concurrent.duration._
class Worker(val host: String, val port: Int, val masterHost: String,
             val masterPort: Int, val memory: Int, val cores: Int) extends Actor{
  // 生成一个Worker ID
  val workerId: String = UUID.randomUUID().toString
  // 用来存储MasterUrl
  var masterUrl: String = _
  // 心跳时间间隔
  val heartbeat_interval: Long = 10000
  // master的Actor
  var master: ActorSelection = _
  override def preStart(): Unit = {
    // 获取Master的Actor
    master = context.actorSelection(s"akka.tcp://${Master.MASTER_SYSTEM}" +
      s"@${masterHost}:${masterPort}/user/${Master.MASTER_ACTOR}")
    //向master发送注册信息
    master ! RegisterWorker(workerId, host, port, memory, cores)
  }

  override def receive = {
    // Worker接收到Master发送过来注册成功的信息（masterUrl）
    case RegisteredWorker(masterUrl) => {
      this.masterUrl = masterUrl
      // 启动一个定时器，定时的给Master发送心跳
      import context.dispatcher
      context.system.scheduler.schedule(
        0 millis, heartbeat_interval millis, self, SendHeartbeat)
    }
    case SendHeartbeat => {
      // 向Master发送心跳信息
      master ! Heartbeat(workerId)
    }
  }
}
object Worker{
  val WORKER_SYSTEM = "WorkerSystem"
  val WORKER_ACTOR = "Worker"

  def main(args: Array[String]): Unit = {
//    val host = args(0) // localhost
//    val port = args(1).toInt
//    val masterHost = args(2)
//    val masterPort = args(3).toInt
//    val memory = args(4).toInt
//    val cores = args(5).toInt
    val host = "127.0.0.1" // localhost
    val port = "8888".toInt
    val masterHost = "127.0.0.1"
    val masterPort = "6666".toInt
    val memory = 16.toInt
    val cores = 64.toInt

    val configStr =
      s"""
         |akka.actor.provider = "akka.remote.RemoteActorRefProvider"
         |akka.remote.netty.tcp.hostname = "$host"
         |akka.remote.netty.tcp.port = "$port"
       """.stripMargin

    // 配置创建Actor需要的配置信息
    val config: Config = ConfigFactory.parseString(configStr)

    // 创建ActorSystem
    val actorSystem: ActorSystem = ActorSystem(WORKER_SYSTEM, config)

    // 用actorSystem实例创建Actor
    actorSystem.actorOf(Props(new Worker(
      host, port, masterHost, masterPort, memory, cores)), WORKER_ACTOR)

    actorSystem.awaitTermination()
  }

}

```


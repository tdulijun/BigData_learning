---

---

# day1

## 回顾

```
基础语言: Java
大数据的生态圈Hadoop 三大组件 HDFS MapReduce Yarn
HDFS用来存储数据 MapReduce用来计算数据 Yarn用来进行资源调度
Zookeeper分布式管理
Hive使用的是类似于SQL语言的HiveQL --> HQL语言
HBase数据库存储海量数据,NoSql-->NO only sql 列式
Flume采集日志
Sqoop数据导出
实时并发的计算框架spark
spark的开发语言:Java,scala,python --> 主要 scala,python
```

## Spark的框架体系 

```
HDFS的优缺点:
优点:
高可靠性,搞扩展性,高效性,高容错性
缺点:
不适合大量小文件存储,不支持多用户写入及任意修改文件,不适合低延迟数据访问
ps:Hadoop2.x之后提供Yarn,进行资源调度
MR是一个离线计算的框架,也就是对以往存储的数据进行分析
Spark是一个实时的计算框架,可以对数据进行一个实时的分析处理
```

## Saprk简介 

```
spark:是一个快如闪电一样的处理引擎
Spark是一种快速、通用、可扩展的大数据分析引擎，2009年诞生于加州大学伯克利分校AMPLab，2010年开源，2013年6月成为Apache孵化项目，2014年2月成为Apache顶级项目。目前，Spark生态系统已经发展成为一个包含多个子项目的集合，其中包含SparkSQL、SparkStreaming、GraphX、MLlib等子项目，Spark是基于内存计算的大数据并行计算框架。Spark基于内存计算，提高了在大数据环境下数据处理的实时性，同时保证了高容错性和高可伸缩性，允许用户将Spark部署在大量廉价硬件之上，形成集群。Spark得到了众多大数据公司的支持，这些公司包括Hortonworks、IBM、Intel、Cloudera、MapR、Pivotal、百度、阿里、腾讯、京东、携程、优酷土豆。当前百度的Spark已应用于凤巢、大搜索、直达号、百度大数据等业务；阿里利用GraphX构建了大规模的图计算和
图挖掘系统，实现了很多生产系统的推荐算法；腾讯Spark集群达到8000台的规模，是当前已知的世界上最大的Spark集群。
```

## 为什么要学习Spark 

```
中间结果输出:基于MapReduce的计算引擎通常会将中结果输出到磁盘上,进行存和容错
处于任务管道承接,考虑当一些查询到MapReduce任务时,往往会产生多个Stage,而串联这些任务，依赖地城HDFS存储系统,每一个Stage会输出一个结果
Spark是MR的替代方案,兼容HDFS Hive并且融到Hadoop生态圈,以弥补MR的不足
```

## Spark特点 

1.快

2.易用

3.通用性

Spark提供了一些通用的解决方案,Spark可以进行批处理,交互式查询,实时流处理,机器学习和图计算 

4.兼容性

spark可以完美的融合其他开源产品
比如:可以完美的融合Hadoop生态圈和Yarn进行融合
Mesos作为资源管理和Spark融合
还可以支持其他的Hadoop组件 

## Spark运行模式 

Local:多用于本地测试,可以在Eclipse或idea中进行程序书写
Standlone:是Spark自带的一个资源调度框架,它完全支持分布式
Yarn:Hadoop生态圈的资源框架,Spark也可以基于yarn来进行计算
Meso:资源调度框架 

## Spark集群安装 

```
基本设置:时间同步,免密,关闭防火墙,安装JDK 集群已经安装完Hadoop集群
1.上传安装包到hadoop01 
2.将文件解压到指定的目录
tar -zxvf spark-1.6.3-bin-hadoop2.6.tgz -C /opt/software/
3.跳转到安装路径进入到conf进行配置
cd /opt/software/spark-1.6.3-bin-hadoop2.6/
cd conf/
3.1修改conf目录下的env文件
mv spark-env.sh.template spark-env.sh
vi spark-env.sh
在文件的末尾添加
export JAVA_HOME=/opt/software/jdk1.7.0_79 JDK安装路径
export SPARK_MASTER_IP=hadoop01 主节点IP
export SPARK_MASTER_PORT=7077 主节点端口号(内部通信)
3.2修改slaves.template文件添加从节点
mv slaves.template slaves
vi slaves
内容:
hadoop02
hadoop03
hadoop04
4.分发配置好的内容到其他节点:
scp -r ./spark-1.6.3-bin-hadoop2.6/ root@hadoop04:$PWD
ps:0后面进行修改 2,3,4
配全局环境变量:
vi /etc/profile
exprot SPARK_HOME=/opt/software/spark-1.6.3-bin-hadoop2.6
需要在引用路径的最后添加 $SPARK_HOME/bin:
保存退出
source /etc/profile
spark启动集群:
进入到安装目录找sbin目录进入 /opt/software/spark-1.6.3-bin-hadoop2.6
启动 ./start-all.sh
sparkt提供webUI界面端
和tomcat的端口是一样 内部通信 7077
```

## Saprk高可用 

```
Master节点存在单点故障，要解决此问题，就要借助zookeeper，并且启动至少两个Master节点来实现高可靠，配置方
式比较简单：
Spark集群规划：hadoop01，hadoop04是Master；hadoop02，hadoop03，hadoop04是Worker
安装配置zk集群，并启动zk集群
停止spark所有服务，修改配置文件spark-env.sh，在该配置文件中删掉SPARK_MASTER_IP并添加如下配置
export SPARK_DAEMON_JAVA_OPTS="-Dspark.deploy.recoveryMode=ZOOKEEPER -
Dspark.deploy.zookeeper.url=hadoop02,hadoop03,hadoop04 -Dspark.deploy.zookeeper.dir=/spark"
分发到hadoop02,hadoop03,hadoop04节点下
1.在hadoop01节点上修改slaves配置文件内容指定worker节点
ps:若是修改了slaves节点那么配置文件也发分发
2.先启动zookeeper集群 ZkServer.sh start
3.在hadoop01上执行sbin/start-all.sh脚本，然后在hadoop04上执行sbin/start-master.sh启动第二个Master
ps:在我们学习的过程中,使用这样的单节点可以跑程序就可以了,在开发环境下使用高可用就行
```

## Spark的程序执行 

若是运行spark任务,需要进入到bin目录下(bin是安装spark目录下的) 

计算π的例子
./spark-submit --class org.apache.spark.examples.SparkPi --master spark://hadoop01:7077
/opt/software/spark-1.6.3-bin-hadoop2.6/lib/spark-examples-1.6.3-hadoop2.6.0.jar 100
可以在webUI界面对任务进行一个监控任务 

在启动任务的时候最先并没有指定资源分配,而是有多少资源就使用多少资源,在跑任务的时候是可以进行资源指定,
指定内存和核心数
--executor-memory 内存大小
--total--executor-cores 核心
在提交任务的时候使用上面这些设置
./spark-submit --class org.apache.spark.examples.SparkPi --master spark://hadoop01:7077 --executor-memory 512m --total-executor-cores 2 /opt/software/spark-1.6.3-bin-hadoop2.6/lib/spark-examples-1.6.3-
hadoop2.6.0.jar 100 

## SparkShell 

spark-shell是spark自带的交互式shell程序
通过shell可以进行交互式编程,可以在shell书写scala语言的spark程序
spark-shell一般是用来进行spark程序测试或联系使用
spark-shell属于Spark的特殊应用程序:
spark-shell启动有两种方式:local模式和cluster模式
spark-shell直接启动就是本地模local模式:
相当于在本机启动一个sparkSubmit进行,没有与集群建立联系,进行中是有Submit,但是不会被提交到集群
Cluster模式(集群模式)
spark-shell --master spark://hadoop1:7077 --executor-memory 512m --total-executor-cores 1 

在使用sparkshell的时候默认就已经创建好了两个变量
Sparkcontext 变量名是 sc
SQLcontext 变量名是 sqlContext
所以在shell 中可以直接使用这两个变量

```scala
sc.textFile("hdfs://hadoop1:8020/word.txt").flatMap(_.split(" ")).map((_,1)).r
educeByKey(_+_).saveAsTextFile("hdfs://hadoop1:8020/out") 
```

单词统计 

## 通过IDEA创建Spark项目 

ps:不在是单纯的scala项目,所以现在需要使用maven来完成项目创建
maven配置:使用的jar包和打包方式 

```xml
<properties>
<maven.compiler.source>1.7</maven.compiler.source>
<maven.compiler.target>1.7</maven.compiler.target>
<encoding>UTF-8</encoding>
<scala.version>2.10.6</scala.version>
<spark.version>1.6.3</spark.version>
<hadoop.version>2.6.4</hadoop.version>
<scala.compat.version>2.10</scala.compat.version>
</properties>
<dependencies>
<dependency>
<groupId>org.scala-lang</groupId>
<artifactId>scala-library</artifactId>
<version>${scala.version}</version>
</dependency><dependency>
<groupId>org.apache.spark</groupId>
<artifactId>spark-core_2.10</artifactId>
<version>${spark.version}</version>
</dependency>
<dependency>
<groupId>org.apache.hadoop</groupId>
<artifactId>hadoop-client</artifactId>
<version>${hadoop.version}</version>
</dependency>
</dependencies>
<build>
<sourceDirectory>src/main/scala</sourceDirectory>
<!--<testSourceDirectory>src/test/scala</testSourceDirectory>-->
<plugins>
<plugin>
<groupId>net.alchim31.maven</groupId>
<artifactId>scala-maven-plugin</artifactId>
<version>3.2.2</version>
<executions>
<execution>
<goals>
<goal>compile</goal>
<goal>testCompile</goal>
</goals>
<configuration>
<args>
<arg>-make:transitive</arg>
<arg>-dependencyfile</arg>
<arg>${project.build.directory}/.scala_dependencies</arg>
</args>
</configuration>
</execution>
</executions>
</plugin>
<plugin>
<groupId>org.apache.maven.plugins</groupId>
<artifactId>maven-shade-plugin</artifactId>
<version>2.4.3</version>
<executions>
<execution>
<phase>package</phase>
<goals>
<goal>shade</goal>
</goals>
<configuration>
<filters>
<filter><artifact>*:*</artifact>
<excludes>
<exclude>META-INF/*.SF</exclude>
<exclude>META-INF/*.DSA</exclude>
<exclude>META-INF/*.RSA</exclude>
</excludes>
</filter>
</filters>
<transformers>
<transformer
implementation="org.apache.maven.plugins.shade.resource.ManifestResourceTransformer">
<mainClass></mainClass>
</transformer>
</transformers>
</configuration>
</execution>
</executions>
</plugin>
</plugins>
</build>
```

```scala
/**
* 用spark实现单词统计
*/
object SparkWordCount {
def main(args: Array[String]): Unit = {
//1.创建sparkConf对象,并设置App名称
/*
这个conf就是之前学习的MapReduce中的conf的配置
通过这个conf对集群中spark-env.sh文件进行修改
setAppName设置集群的名字,不设置默认就是UUID所产生的一个名字
setMaster设置运行模式,不写默认认为是提交集群
setMaster("local[决定使用多少个core]") 设置本地模式
core值不是核心可以当做有多少个线程数, 什么都不写或写* 那么久相当于
系统中有多个空闲的线程就使用多少个线程
*/
val conf = new SparkConf().setAppName("SparkWordCount")
//2.创建SparkContext 提交SparkApp的入口
val sc = new SparkContext(conf)
/**
* sc所调用的方法叫做是算子,算子有两种
* transformation(只能计算没有结果) 和 action (触发得到结果)
*/
//3.获取HDFS上的数据
val lines: RDD[String] = sc.textFile(args(0))
//4.将数据进行切分并压平
val words: RDD[String] = lines.flatMap(_.split(" "))
//5.遍历当前数据组组成二元组(key,1)
val tuples: RDD[(String, Int)] = words.map((_,1))
//6.进行聚合操作,相同key 的value进行相加
val sumed: RDD[(String, Int)] = tuples.reduceByKey(_+_,1)本地运行:
val sorted: RDD[(String, Int)] = sum1.sortBy(_._2,false)
第一个修改的位置:
1.在创建conf对象要设置setMaster("local")
读取数据
1.直接在工程读取
sc.textFile("相对路径 或者 绝对路径")
2.从HDFS上获取数据(集群开启)
ps:端口号是内部通信端口号:8020 / 9000
sc.textFile("hdfs://集群:端口/文件名")
写出数据
直接打印到控制台上或者直存储到本地
sorted.saveAsTextFile("out") 本地存路径
打印数据
println(sorted.collect.toBuffer)
sorted.foreach(x => println(x))
//7.可以对数据进排序操作,有默认参数true升序,降序 false就可以刻
val sorted: RDD[(String, Int)] = sumed.sortBy(_._2,false)
//8.将数据存储到HDFS上
val unit: Unit = sorted.saveAsTextFile(args(1))
//9.回收资源停止sc结束任务
sc.stop()
}
} 
提交:
spark-submit \
--class SparkWordCount \
--master spark://hadoop01:7077 \
--executor-memory 512m \
--total-executor-cores 3 \
/root/Spark-1.0-SNAPSHOT.jar hdfs://hadoop01:8020/word.txt hdfs://hadoop01:8020/out
```

## 本地运行: 

```
第一个修改的位置:
1.在创建conf对象要设置setMaster("local")
读取数据
1.直接在工程读取
sc.textFile("相对路径 或者 绝对路径")
2.从HDFS上获取数据(集群开启)
ps:端口号是内部通信端口号:8020 / 9000
sc.textFile("hdfs://集群:端口/文件名")
写出数据
直接打印到控制台上或者直存储到本地
sorted.saveAsTextFile("out") 本地存路径
打印数据
println(sorted.collect.toBuffer)
sorted.foreach(x => println(x))
sorted.foreach(println)
```

# day2

## Java版本的WordCount 

```java
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;
import java.util.Arrays;
import java.util.List;
/**
* java版本wordCount
*/
public class JavaWordCount {
public static void main(String[] args) {
//1.先创建conf对象进行配置主要是设置名称,为了设置运行模式
SparkConf conf = new SparkConf().setAppName("JavaWordCount").setMaster("local");
//2.创建context对象
JavaSparkContext jsc = new JavaSparkContext(conf);
JavaRDD<String> lines = jsc.textFile("dir/File.txt");
//进行切分数据 flatMapFunction是具体实现类
JavaRDD<String> words = lines.flatMap(new FlatMapFunction<String, String>() {
//Iterable 是所有集合的超级父接口
@Override
public Iterable<String> call(String s) throws Exception {
List<String> splited = Arrays.asList(s.split(" "));
return splited;
}
});
//将数据生成元组
//第一个泛型是输入的数据类型 后两个参数是输出参数元组的数据
JavaPairRDD<String, Integer> tuples = words.mapToPair(new PairFunction<String, String,
Integer>() {
@Override
public Tuple2<String, Integer> call(String s) throws Exception {
return new Tuple2<String, Integer>(s, 1);
}
});
//聚合
JavaPairRDD<String, Integer> sumed = tuples.reduceByKey(new Function2<Integer, Integer,
Integer>() {
@Override
//第一个Integer是相同key对应的value
//第二个Integer是相同key 对应的value
public Integer call(Integer v1, Integer v2) throws Exception {
return v1 + v2;
}
});
//因为Java api没有提供sortBy算子,此时需要将元组中的数据进行位置调换,然后在排序,排完序在换回
//第一次交换是为了排序
JavaPairRDD<Integer, String> swaped = sumed.mapToPair(new PairFunction<Tuple2<String,
Integer>, Integer, String>() {
@Override
public Tuple2<Integer, String> call(Tuple2<String, Integer> tup) throws Exception {
return tup.swap();
}
});
//排序
JavaPairRDD<Integer, String> sorted = swaped.sortByKey(false);
//第二次交换是为了最终结果 <单词,数量>
JavaPairRDD<String, Integer> res = sorted.mapToPair(new PairFunction<Tuple2<Integer,
String>, String, Integer>() {
@Override
public Tuple2<String, Integer> call(Tuple2<Integer, String> tuple2) throws Exception
{
return tuple2.swap();
}
});
System.out.println(res.collect());
res.saveAsTextFile("out");
jsc.stop();
}
}
```

## 什么是RDD 

Spark是一个大数据分布式并行计算框架,不仅是实现了MapReduce的算子map函数和reduce函数形成了一个计算模型,还提供了更加丰富的算子,Spark中提的算子概念就可以简称为RDD

RDD叫做分布式数据集,是Spark中最基本的数据抽象,它是一个**不可变,可分区,里面的元素可以并行计算的集合,**RDD具有数据流模型的特点:自动容错,位置感知和可伸缩,RDD允许在执行多换个查询时显示的将工作集缓存在内存中,后续的查询能够重用这个工作集,这样可以提高效率
ps:
Spark和很多其他分布式计算系统都借用了"分而治之"思想来实现并行处理:
把一个超大的数据集,切分成N个小堆,找M个执行器,各自拿一块或多块数据执行,执行出结果后在在进行会后,spark的工作就是:凡是能被我处理的,都要符合我的严要求,所以spark无论在处理什么数据之前都会将数据多块,存这个多
块数据的数据集就是RDD 

RDD就像操作本地集合集合一样,有很多的方法可以调用(算子),使用方便,无序关系底层实现val lines:RDD[String] = sc.textFile("路径")通过这个方法(算子)可以返回的数据类型是RDD,可以想成通过testFile这个方法获取文件中的数据然后存储到RDD类型中深入一些实际RDD是一个引用(指针),引用这个文件,这个文件中的一些信息可以通过这个引用来进行操作 

## 详细的RDD特征 

```
RDD的属性
1）一组分片（Partition），即数据集的基本组成单位。对于RDD来说，每个分片都会被一个计算任务处理，并决定并行计算的粒度。用户可以在创建RDD时指定RDD的分片个数，如果没有指定，那么就会采用默认值。默认值就是程序所分配到的CPU Core的数目。
2）一个计算每个分区的函数。Spark中RDD的计算是以分片为单位的，每个RDD都会实现compute函数以达到这个目的。compute函数会对迭代器进行复合，不需要保存每次计算的结果。
3）RDD之间的依赖关系。RDD的每次转换都会生成一个新的RDD，所以RDD之间就会形成类似于流水线一样的前后依赖关系。在部分分区数据丢失时，Spark可以通过这个依赖关系重新计算丢失的分区数据，而不是对RDD的所有分区进行重新计算。
4）一个Partitioner(分区器)，即RDD的分片函数。当前Spark中实现了两种类型的分片函数，一个是基于哈希的HashPartitioner，另外一个是基于范围的RangePartitioner。只有对于key-value的RDD，才会有Partitioner，非key-value的RDD的Parititioner的值是None。Partitioner函数不但决定了RDD本身的分片数量，也决定了parent RDD shuffle输出时的分片数量。
5）一个列表，存储存取每个Partition的优先位置（preferred location）。对于一个HDFS文件来说，这个列表保存的就是每个Partition所在的块的位置。按照“移动数据不如移动计算”的理念，Spark在进行任务调度的时候，会尽可能地将计算任务分配到其所要处理数据块的存储位置。
```

## RDD算子 

RDD算子可以分为两种类型Transformation(转换)
RDD中的所谓转换都是延迟加载,也就是,它们不会直接计算结果,相反它们只
是几组这些用应用到基础数据集上的转换动作,只有当发生一个要求返回结果
给Driver的动作时,这些转换才会真正的运行这样的设计可以让spark更加有
效率的运行 

| 转换                                                 | 含义                                                         |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| map(func)                                            | 返回一个新的RDD，该RDD由每一个输入元素经 过func函数转换后组成 |
| filter(func)                                         | 返回一个新的RDD，该RDD由经过func函数计算 后返回值为true的输入元素组成 |
| flatMap(func)                                        | 类似于map，但是每一个输入元素可以被映射为0 或多个输出元素（所以func应该返回一个序列， 而不是单一元素） |
| mapPartitions(func)                                  | 类似于map，但独立地在RDD的每一个分片上运 行，因此在类型为T的RDD上运行时，func的函数 类型必须是Iterator[T] => Iterator[U] |
| mapPartitionsWithIndex(func)                         | 类似于mapPartitions，但func带有一个整数参数 表示分片的索引值，因此在类型为T的RDD上运行 时，func的函数类型必须是(Int, Iterator[T]) => Iterator[U] |
| sample(withReplacement, fraction, seed)              | 根据fraction指定的比例对数据进行采样，可以选 择是否使用随机数进行替换，seed用于指定随机 数生成器种子 |
| union(otherDataset)                                  | 对源RDD和参数RDD求并集后返回一个新的RDD                      |
| intersection(otherDataset)                           | 对源RDD和参数RDD求交集后返回一个新的RDD                      |
| distinct([numTasks]))                                | 对源RDD进行去重后返回一个新的RDD                             |
| groupByKey([numTasks])                               | 在一个(K,V)的RDD上调用，返回一个(K, Iterator[V])的RDD        |
| reduceByKey(func, [numTasks])                        | 在一个(K,V)的RDD上调用，返回一个(K,V)的 RDD，使用指定的reduce函数，将相同key的值聚 合到一起，与groupByKey类似，reduce任务的个 数可以通过第二个可选的参数来设置 |
| aggregateByKey(zeroValue)(seqOp, combOp, [numTasks]) | 相同的Key值进行聚合操作，在聚合过程中同样使 用了一个中立的初始值zeroValue:中立值,定义返 回value的类型，并参与运算seqOp:用来在同一个 partition中合并值combOp:用来在不同partiton 中合并值 |
| sortByKey([ascending], [numTasks])                   | 在一个(K,V)的RDD上调用，K必须实现Ordered接 口，返回一个按照key进行排序的(K,V)的RDD |
| sortBy(func,[ascending], [numTasks])                 | 与sortByKey类似，但是更灵活                                  |
| join(otherDataset, [numTasks])                       | 在类型为(K,V)和(K,W)的RDD上调用，返回一个相 同key对应的所有元素对在一起的(K,(V,W))的RDD |

| 转换                                            | 含义                                                         |
| ----------------------------------------------- | ------------------------------------------------------------ |
| cogroup(otherDataset, [numTasks])               | 在类型为(K,V)和(K,W)的RDD上调用，返回一个(K, (Iterable,Iterable))类型的RDD |
| cartesian(otherDataset)                         | 笛卡尔积                                                     |
| pipe(command, [envVars])                        | 将一些shell命令用于Spark中生成新的RDD                        |
| coalesce(numPartitions)                         | 重新分区                                                     |
| repartition(numPartitions)                      | 重新分区                                                     |
| repartitionAndSortWithinPartitions(partitioner) | 重新分区和排序                                               |

## Action(动作):在RDD上运行计算,并返回结果给Driver或写入文件系统 

| 动作                                    | 含义                                                         |
| --------------------------------------- | ------------------------------------------------------------ |
| reduce(func)                            | 通过func函数聚集RDD中的所有元素，这个功能必须是可交换且 可并联的 |
| collect()                               | 在驱动程序中，以数组的形式返回数据集的所有元素               |
| count()                                 | 返回RDD的元素个数                                            |
| first()                                 | 返回RDD的第一个元素（类似于take(1)）                         |
| take(n)                                 | 返回一个由数据集的前n个元素组成的数组                        |
| takeSample(withReplacement,num, [seed]) | 返回一个数组，该数组由从数据集中随机采样的num个元素组 成，可以选择是否用随机数替换不足的部分，seed用于指定随机 数生成器种子 |
| takeOrdered(n, [ordering])              | takeOrdered和top类似，只不过以和top相反的顺序返回元素        |
| saveAsTextFile(path)                    | 将数据集的元素以textfile的形式保存到HDFS文件系统或者其他支 持的文件系统，对于每个元素，Spark将会调用toString方法，将 它装换为文件中的文本 |
| saveAsSequenceFile(path)                | 将数据集中的元素以Hadoop sequencefile的格式保存到指定的目 录下，可以使HDFS或者其他Hadoop支持的文件系统。 |
| saveAsObjectFile(path)                  |                                                              |
| countByKey()                            | 针对(K,V)类型的RDD，返回一个(K,Int)的map，表示每一个key对 应的元素个数。 |
| foreach(func)                           | 在数据集的每一个元素上，运行函数func进行更新。               |

ps:1个Action相当于是一个Job
Transformation属于延迟计算,当一个RDD转换另一个RDD的时候并没有立即转换,仅仅是记住了数据集的逻辑操作,
当Action触发Spark作业的时候,才会真正的执行 

## 算子简单使用: 

```scala
package Day02
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
object PrimaryOperatorDemo {
def main(args: Array[String]): Unit = {
//因为spark-shell 可以分为两种运行模式
//集群模式 spark-shell --master spark://hadoop01:7077
//--executor-memory 内存m --toatl-executor-cores 核数
//本地模式 spark-shell
//无论是本地还是集群这列都已经创建好了两个变量
//sparkcontext -- >sc sqlcontext --> sqlcontext
//在Idea中需要自己创建sc对象并且指定运行模式
val conf = new SparkConf().setAppName("PrimaryOperatorDemo").setMaster("local")
val sc = new SparkContext(conf)
//通过并行化生成rdd
val rdd = sc.parallelize(List(5,6,4,7,3,8,2,9,10))
//对rdd里面每一个元乘以2然后排序
// sortBy 第一个参数是遍历, 第二个参数是排序方式 true是升序
val rdd2: RDD[Int] = rdd.map(_ * 2).sortBy(x => x, true)
println(rdd2.collect().toBuffer)
//会受到线程影响
//rdd3.foreach(x =>print(x+" "))
//过滤出结果大于10
val rdd3: RDD[Int] = rdd2.filter(_ > 10)
println(rdd3.collect().toBuffer)
val rdd4 = sc.parallelize(Array("a b c","b c d"))
//将rdd4中的元素进行切分后压平
val rdd5: RDD[String] = rdd4.flatMap(_.split(" "))
println(rdd5.collect().toBuffer)
//假如: List(List(" a,b" ,"b c"),List("e c"," i o"))
//压平 flatMap(_.flatMap(_.split(" ")))
//求并集
val rdd6 = sc.parallelize(List(5,6,7,8))
val rdd7 = sc.parallelize(List(1,2,5,6))
val rdd8 = rdd6 union rdd7
println(rdd8.collect.toBuffer)
//求交集
val rdd9 = rdd6 intersection rdd7 println(rdd9.collect.toBuffer)
//去重出重复
println(rdd8.distinct.collect.toBuffer)
//join
val rdd10_1 = sc.parallelize(List(("tom",1),("jerry" ,3),("kitty",2)))
val rdd10_2 = sc.parallelize(List(("jerry" ,2),("tom",2),("dog",10)))
//相同的key会被合并
val rdd10_3 = rdd10_1 join rdd10_2
println(rdd10_3.collect().toBuffer)
//左连接和右连接
//除基准值外是Option类型,因为可能存在空值所以使用Option
val rdd10_4 = rdd10_1 leftOuterJoin rdd10_2 //以左边为基准没有是null
val rdd10_5 = rdd10_1 rightOuterJoin rdd10_2 //以右边为基准没有是null
println(rdd10_4.collect().toList)
println(rdd10_5.collect().toBuffer)
//求一个并集
val rdd11: RDD[(String, Int)] = rdd10_1 union rdd10_2
println(rdd11.collect().toList)
//按照key进行分组
val rdd11_1 = rdd11.groupBy(_._1)
println(rdd11_1.collect().toBuffer)
//按照key进行分组,并且可以制定分区
val rdd11_1_1 = rdd11.groupByKey
println(rdd11_1_1.collect().toList)
//cogroup合并数据并根据相同key进行排序
//cogroup和groupByKey的区别
//cogroup输入的数据必须是(k,v)和另外一个(k,w) 得到一个(k,(seq[v],seq[w]))的数据集
//groupByKey:进行对已经合并好的数据根据相同key进行分组 得到一个(k,seq[v])
//分组的话需要提供二元组(k,v)
val rdd12_1 = sc.parallelize(List(("tom",1),("jerry" ,3),("kitty",2)))
val rdd12_2 = sc.parallelize(List(("jerry" ,2),("tom",2),("dog",10)))
val rdd12_3: RDD[(String, (Iterable[Int], Iterable[Int]))] = rdd12_1.cogroup(rdd12_2)
println(rdd12_3.collect.toBuffer)
//相同key的value进行计算
val rdd13_2 = rdd11.reduceByKey((x,y)=>x+y)
println(rdd13_2.collect().toList)
//求和
val rdd14 = sc.parallelize(List(1,2,3,5,6))
val rdd14_1 = rdd14.reduce(_+_)
//需求:List(("tom",1),("jerry" ,3),("kitty",2) List(("jerry" ,2),("tom",2),("dog",10))
//要先合并数据 ,按key进行聚合,根据key进行排序(根据数值而非字符串)
val rdd15_1 = sc.parallelize(List(("tom",1),("jerry" ,3),("kitty",2)))
val rdd15_2 = sc.parallelize(List(("jerry" ,2),("tom",2),("dog",10)))
//合并数据TextFile Partition个数
val rdd15_3 = rdd15_1 union rdd15_2
//聚合
val rdd15_4: RDD[(String, Int)] = rdd15_3.reduceByKey(_ + _)
//排序 sortBykey
//现在数据key是String类型,后面Int类型进行排序
val rdd15_5: RDD[(Int, String)] = rdd15_4.map(t => (t._2,t._1))
val rdd15_6: RDD[(Int, String)] = rdd15_5.sortByKey(false)
val rdd15_7: RDD[(String,Int)] = rdd15_6.map(t => (t._2,t._1))
println(rdd15_7.collect().toBuffer)
//笛卡尔积
val rdd16 = rdd15_1 cartesian rdd15_2
val rdd17 = sc.parallelize(List(2,5,1,62,7,3,267))
println(rdd17.count())//数据个数
println(rdd17.top(3).toBuffer) //取值 默认会降序排序, 若输入0 会返回一个空数组
println(rdd17.take(3).toBuffer) //取值 取出对应数量的数值
println(rdd17.takeOrdered(3).toBuffer)//取值会进行排序,默认是升序 返回对等数量的数据
println(rdd17.first()) //获取第一个值
}
}
```

## TextFile Partition个数 

```scala
package Day02
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapred.TextInputFormat
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
object TextFileDemo {
def main(args: Array[String]): Unit = {
val conf = new SparkConf().setAppName("PrimaryOperatorDemo").setMaster("local[*]")
val sc = new SparkContext(conf)
//通过并行化生成rdd
// //指定多少分区就会的到多少分区
// val rdd = sc.parallelize(List(5,6,4,7,3,8,2,9,10),10)
// println(rdd.partitions.length)//获取分区个数
//可以指定分区数量,这个分区数量默认就是2,但是loacl中默认值小于2就会使用这个默认值
//指定分区个数,具体分区数需要通过文件大小和传入分区分区数进行计算,最终才能确定
val rdd1 = sc.textFile("hdfs://hadoop01:8020/word.txt",4).flatMap(_.split("
")).map((_,1)).reduceByKey(_+_)
println(rdd1.partitions.length)//获取分区个数
/**
* 总结:
* textFile在没有指定分区数的情况下默认是2,除非指定值小于2否则就会使用默认分区
* textFile指定分去了,根据提供的分区和文件的大小来计算切片个数,这个切片的个数就是具体点partition的
个数
*/
}
}
```

## 算子进阶 

```scala
package Day02
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
/*
进阶算子
*/
object PlusOperatorDemo {
//自定义打印方法
def printlns[T](rdd:RDD[T]): Unit ={
println(rdd.collect.toBuffer);
} 
def main(args: Array[String]): Unit = {
//遍历出集合中每一个元素
val conf = new SparkConf().setAppName("PrimaryOperatorDemo").setMaster("local")
val sc = new SparkContext(conf)
val rdd1 = sc.parallelize(List(1,2,3,4,5,6),3)
val rdd2:RDD[Int] = rdd1.map(_ * 10)
//printlns(rdd2)
//mapPartitions是对每个分区中的数据进行迭代
//第一个参数是第一个迭代器送对象, 第二个参数表示是否保留父RDD的partition分区信息
//第二个参数的值默认是false一般是不修改(父RDD就需要使用到宽窄依赖的问题)
//(f: Iterator[T] => Iterator[U],preservesPartitioning: Boolean = false)
//ps:第一个_是迭代器对象 第二个_是分区(集合)中数据作业:aggregate的使用
val rdd3: RDD[Int] = rdd1.mapPartitions(_.map(_*10))
/*
ps:如果RDD数据量不大,建议采用mapPartition算子代替map算子,可以加快数据量的处理数据
但是如果RDD中数据量过大比如10亿条,不建议使用mapPartitions开进行数据遍历,可能出现oom内存溢出错误
*/
//mapwith:是对Rdd中每一个元素进行操作(是map方法的另外一种方法)
//计算每一个分区中的值
//(constructA: Int => A, preservesPartitioning: Boolean = false)(f: (T, A) => U)
//(1,2,3,4,5,6)
//参数是柯里化
//第一个参数是分区号(分区好是从0开始) 第二个参数是否保留父RDD的partition分析信息 一,二参数是一个
//第二个参数T分区中的每一个元素 A是第一个柯里化参数中第一参数得到的结果
//柯里化第一参数是分区号逻辑 柯里化第二参数是实际对应分区中元素的处理逻辑
//这个方法已经过时
val rdd4: RDD[Int] = rdd1.mapWith(i=>i*10)((a, b)=>a+b+2)
// printlns(rdd4)
//flatMapWith是对rdd中每一个元素进行操作返回的结果是一个序列数据是扁平化处理过后的
//(constructA: Int => A, preservesPartitioning: Boolean = false)(f: (T, A) => Seq[U])
//参数使用和上面是一样的只不过最后的返回值是还是一个序列
//这个方法也过时
val rdd5: RDD[Int] = rdd1.flatMapWith(i=>i)((x, y)=>List(y,x))
//printlns(rdd5)
//mapPartitionsWithIndex 是对rdd中每个分区的遍历出操作
//(f: (Int, Iterator[T]) => Iterator[U],preservesPartitioning: Boolean = false)
//参数是一个柯里化 第二个参数是一个隐式转换
//函数的作用和mapPartitions是类似,不过要提供两个采纳数,第一个参数是分区号
//第二个参数是分区中元素存储到Iterator中(就可以操作这个Iterator)
//第三个参数是否保留符RDD的partitoin
val func = (index:Int,iter:Iterator[(Int)]) =>{
iter.map(x => "[partID:"+index + ", value:"+x+"]")
}
    val rdd6: RDD[String] = rdd1.mapPartitionsWithIndex(func)
printlns(rdd6)
}
}
```

# day3

## 进阶算子 Aggregate: 

```scala
package Day03
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
/*
Aggregate算子 :聚合
Aggregate函数将每个分区里面的元素进行聚合,然后用combine函数将每个分区的结果和初始值进行combine操作
ps:combine --> 执行一次reduce,这个函数最终的返回值类型不要和RDD中元素的类型一致
zeroValue是初始值(默认值) seqOP局部聚合(分区) combOp 全局聚合
(zeroValue: U)(seqOp: (U, T) => U, combOp: (U, U) => U)
*/
object AggregateDemo {
def fun1[T](index:Int,iter:Iterator[(T)]): Iterator[String] ={
iter.map(x => "[partID:"+index+" value:"+x+"]")
} 
    def main(args: Array[String]): Unit = {
val conf = new SparkConf().setAppName("SparkWordCount").setMaster("local[2]")
val sc = new SparkContext(conf)
val rdd1 = sc.parallelize(List(1,2,3,4,5,6,7,8,9),2)
println(rdd1.mapPartitionsWithIndex(fun1).collect.toBuffer)
//zeroValue是初始值(默认值) seqOP局部聚合(分区) combOp 全局聚合
/*
初始值会先和分区中的数据进行比较计算,最终全局聚合的时候每个分区值相加然后在加上默认值
ps:初始值会跟分区进行局部聚合(分区计算)
初始值会跟全局聚合在一次进行计算
aggreagte是action类型的算子
*/
val sumed: Int = rdd1.aggregate(0)(math.max(_,_),_+_)
println(sumed)
val sumed2: Int = rdd1.aggregate(5)(math.max(_,_),_+_)
println(sumed2)
println("-------------------------------------华丽的分割线-----------------------------------
-------")
val rdd2 = sc.parallelize(List("a","b","c","d" ,"e","f"),2)
println(rdd2.mapPartitionsWithIndex(fun1).collect.toBuffer)
// abcefg 或 defabc 为什么?
//aggregate是先计算分区的值,并行化处理的情况
//2个分区可能出现两个线程在跑,那个分区先完成不一定,所以就出现谁先执行完谁就在前面,剩下的就在后面
val str: String = rdd2.aggregate("")(_+_,_+_)
println(str)
/*
初始值在进行局部聚合的时候会和分区中的值进行一次计算
所有分区计算完成后会在全局聚合的时候在进行一次计算
*/
val str1: String = rdd2.aggregate("=")(_+_,_+_)
println(str1)val rdd3 = sc.parallelize(List("12","34","345","4567"),2)
println("-------------------------------------华丽的分割线-----------------------------------
-------")
println(rdd3.mapPartitionsWithIndex(fun1).collect.toBuffer)
//24 或 42 ""+2+4 或 "" +4+2
val str2: String = rdd3.aggregate("")((x, y)=>math.max(x.length,y.length).toString, (x,
y)=>x+y)
println(str2)
val rdd4 = sc.parallelize(List("12","34","345",""),2)
println("-------------------------------------华丽的分割线-----------------------------------
-------")
println(rdd4.mapPartitionsWithIndex(fun1).collect.toBuffer)
// 10 或01 ""+1+0 ""+0+1
println("".length)
val str3: String = rdd4.aggregate("")(
(x, y)=>{math.min(x.length,y.length).toString},
(x, y)=> {x + y}
)
println(str3)
val rdd5 = sc.parallelize(List("12","34","","345"),2)
println("-------------------------------------华丽的分割线-----------------------------------
-------")
println(rdd4.mapPartitionsWithIndex(fun1).collect.toBuffer)
println("".length)
val str4: String = rdd5.aggregate("")(
(x, y)=>{math.min(x.length,y.length).toString},
(x, y)=> {x + y}
) 
println(str4)
//AggregateByKey
//相同key中的值进行聚合操作,通过AggregateBykey函数最终返回值的类型还是RDD(PairRDD)
val rdd6 = sc.parallelize(List(("cat",2),("cat",5),("pig",10),("dog",3),("dog",4),
("cat",4)),2)
println("-------------------------------------华丽的分割线-----------------------------------
-------")
println(rdd6.mapPartitionsWithIndex(fun1).collect.toBuffer)
//AggregateBykey对应的是二元组的计算,使用方式和Aggregate没有太大的区别
//初始值 分区聚合(局部) 全局聚合
//先计算对应分区的值,然后全局聚合
//ps:因为第一次给的数值每个分区中是没有相同key所有都是最大值,所有就相当于值都值相加了
// 第二次将同一个分区中的key有相同
//首先会根据相同key来进行计算,以cat为例先会和初始值-进行计算比较留下最大值
//然后会的等待第二分区完成计算,然后在进行一个全局的聚合
val value: RDD[(String, Int)] = rdd6.aggregateByKey(0)(math.max(_,_),_+_)
println(value.collect.toBuffer)
}
}
```

## 算子combineByKey 

```scala
package Day03
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scala.swing.event.AdjustingEvent
//根据相同key进行聚合
/*
(createCombiner: V => C,mergeValue: (C, V) => C,mergeCombiners: (C, C) => C)
第一个参数:是遍历集合中value的值 V和C的数据类型由value的数据类型决定
ps:相同于根据key 生成一个类型 key [多value]的集合 例如: (hello list(1,1,1,1,1,1,1,1,1))
第二个参数 局部聚合
c的数据类型是第一个参数的返回值决定
v的数据类型是第一个参数的,参数数据类型决定
ps: 若 key [多个value]集合,进行局部的聚合 若没有就不进入局部计算
第三个参数 全局聚合
对第二个函数中每个分区操作产生的记过,在次进行聚合
C的数据类型是第二个函数得到的数据类型 最终聚合
*/
object combineByKey {
def fun1[T](index:Int,iter:Iterator[(T)]): Iterator[String] ={
iter.map(x => "[partID:"+index+" value:"+x+"]")
} 
def main(args: Array[String]): Unit = {
val conf = new SparkConf().setAppName("SparkWordCount").setMaster("local[2]")
val sc = new SparkContext(conf)
val rdd1: RDD[(String, Int)] = sc.textFile("dir/File.txt").flatMap(_.split(" ")).map((_,1))
val rdd2: RDD[(String, Int)] = rdd1.combineByKey(x => x, (a:Int, b:Int)=>a+b, (m:Int,
n:Int)=> m + n)
println(rdd2.collect.toBuffer)
//若第一个采纳数进行计算,相当是对分区中的数据进行一次计算,这个值不会传在全局聚合中
//可以简单的理解为是AggregateByKey中的初始值,但是这个初始值参加 全局聚合
val rdd3: RDD[(String, Int)] = rdd1.combineByKey(x=>x+10 , (a:Int, b:Int)=>a+b, (m:Int,
n:Int)=> m + n)
println(rdd3.collect.toBuffer)
println("====================================================================================")
val rdd4 =其他算子
sc.parallelize(List("tom","jerry","kitty","cat","dog","pig","bird","bee","wolf"),3)
val rdd5 = sc.parallelize(List(1,1,2,2,2,1,2,2,2),3)
val rdd6:RDD[(Int,String)] = rdd5.zip(rdd4)
println(rdd6.mapPartitionsWithIndex(fun1).collect.toBuffer)
//需求key是1的放到一起 key是2的一起
//第一个参数是获取vlue的值
val rdd7 = rdd6.combineByKey(x=>List(x),(x:List[String],y:String)=> x :+ y,
(m:List[String],n:List[String]) => m ++ n)
println(rdd7.collect.toBuffer)
}
}
```

## 其他算子 

```scala
package Day03
import org.apache.spark.rdd.RDD
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
object OtherOperatorDemo {
def fun1[T](index:Int,iter:Iterator[(T)]): Iterator[String] ={
iter.map(x => "[partID:"+index+" value:"+x+"]")
} 
  def main(args: Array[String]): Unit = {
val conf = new SparkConf().setAppName("SparkWordCount").setMaster("local")
val sc = new SparkContext(conf)
val rdd1 = sc.parallelize(List(("a",1),("b",1),("a",1)))
//countByKey 属于action类型的算子 统计key的个数
val key: collection.Map[String, Long] = rdd1.countByKey
println(key)
//统计value的个数 但是会将集合中的一个元素看做是一个vluae
val value: collection.Map[(String, Int), Long] = rdd1.countByValue
println(value)
//filterByRange:对RDD中的元素进行过滤,返回指定范围内的数据
val rdd2 = sc.parallelize(List(("e",5),("c",3),("d",4),("c",2),("a",1)))
val rdd2_1: RDD[(String, Int)] = rdd2.filterByRange("c","e")//包括开始和结束的
println(rdd2_1.collect.toList)
//flatMapValues对参数进行扁平化操作,是value的值
val rdd3 = sc.parallelize(List(("a","1 2"),("b","3 4")))
println( rdd3.flatMapValues(_.split(" ")).collect.toList)
   输出结果 List((a,1), (a,2), (b,3), (b,4))
//foldByKey 根据相同key进行聚合
// val rdd4 = sc.textFile("dir/File.txt").flatMap(_.split(" ")).map((_,1)).foldByKey(0)(_+_)
// println(rdd4.collect().toList)
//foreachPartition 循环的是分区数据val rdd5 = sc.parallelize(List(1,2,3,4,5,6,7,8,9),3)
//rdd5.foreachPartition(x => println(x.toList))
rdd5.foreachPartition(x => println(x.reduce(_+_)))
// foreachPartiton一般应用于数据的持久化,存入数据库,可以进行分区的数据存储
//keyBy 以传入的函数返回值作为key ,RDD中的元素为value 新的元组
val rdd6 = sc.parallelize(List("dog","cat","pig","wolf","bee"),3)
val rdd6_1: RDD[(Int, String)] = rdd6.keyBy(_.length)
println(rdd6_1.collect.toList)
//keys获取所有的key values 获取所有的values
println(rdd6_1.keys.collect.toList)
println(rdd6_1.values.collect.toList)
//collectAsMap 将需要的二元组转换成Map
val map: collection.Map[String, Int] = rdd2.collectAsMap()
println(map)
//重新分区算子:
//repatition coalesce partitionBy
val rdd7 = sc.parallelize(1 to 10, 4)
println(rdd7.mapPartitionsWithIndex(fun1).collect().toList)
println(rdd7.partitions.length)
//重新分区数据会进行shuffle
val rdd7_1: RDD[Int] = rdd7.repartition(6)
println(rdd7_1.mapPartitionsWithIndex(fun1).collect().toList)
println(rdd7_1.partitions.length)
//第二参数是shuffle 默认不是false
//当前分区数大于原因有分区数若不shuffle 不会进行修改,只有改变为true才会
//当前当前分区数小于原有分区数会直接分区,false不shuffle true可以shuffle
val rdd7_2: RDD[Int] = rdd7.coalesce(3,true)
println(rdd7_2.mapPartitionsWithIndex(fun1).collect().toList)
println(rdd7_2.partitions.length)
//partitionBy 必须是kv数据类型
val rdd7_3 = sc.parallelize(List(("e",5),("c",3),("d",4),("c",2),("a",1)),2)
//可以传入自定分区器, 也可以传入默认分区器 HashPartitioner
val rdd7_4: RDD[(String, Int)] = rdd7_3.partitionBy(new HashPartitioner(4))
println(rdd7_4.partitions.length)
//Checkpoint
/*
检查点,类似于快照,chekpoint的作用就是将DAG中比较重要的数据做一个检查点,将结果存储到一个高可用的地方法
*/
//1.指定存储目录
sc.setCheckpointDir("hdfs://hadoop01:8020/ck")
val rdd8 = sc.textFile("hdfs://hadoop01:8020/word.txt").flatMap(_.split("
")).map((_,1)).reduceByKey(_+_)
//检查点的触发一定要使用个action算子Spark的启动流程和任务提交
rdd8.checkpoint()
rdd8.saveAsTextFile("hdfs://hadoop01:8020/out10")
println(rdd8.getCheckpointFile) //查看存储的位置
//查看是否可以设置检查点 rdd8.isCheckpointed
}
}
```

## Spark的启动流程和任务提交 

![图像 1](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 1.png)

# day4

## 案例:基站停留时间TopN 

```scala
package Day04
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
/*
根据用户产生日志的信息,在那个基站停留时间最长
19735E1C66.log 这个文件中存储着日志信息
文件组成:手机号,时间戳,基站ID 连接状态(1连接 0断开)
lac_info.txt 这个文件中存储基站信息
文件组成 基站ID, 经,纬度
在一定时间范围内,求所用户经过的所有基站所停留时间最长的Top2
思路:
1.获取用户产生的日志信息并切分
2.用户在基站停留的总时长
3.获取基站的基础信息
4.把经纬度的信息join到用户数据中
5.求出用户在某些基站停留的时间top2
*/
object BaseStationDemo {
def main(args: Array[String]): Unit = {
//1.创建SparkConf对象进行配置,然后在创建SparkContext进行操作
val conf = new SparkConf().setAppName("BaseStationDemo").setMaster("local[2]")
val sc = new SparkContext(conf)
//2.获取用户访问基站的log日志
//ps:因为使用是绝对路径所以,你在使用时候需要修改这个路径
val files: RDD[String] = sc.textFile("C:\\Users\\Administrator\\Desktop\\lacduration\\log")
//3.切分用户log日志
val userInfo: RDD[((String, String), Long)] = files.map(line => {
val fields: Array[String] = line.split(",") //切分
val phone = fields(0) //用户手机号
val time = fields(1).toLong //时间戳(数字)
val lac = fields(2) //基站ID
val eventType = fields(3)
//时间类型(连接后断开)
//连接时长需要进行一个区分,因为进入基站范围内有两种状态,这个状态决定时间的开始于结束
val time_long = if (eventType.equals("1")) -time else time
//元组 手机号和基站作为key 时间作为value
((phone, lac), time_long)
})
//用户在相同基站所停留总时长
val sumed: RDD[((String, String), Long)] = userInfo.reduceByKey(_+_)
//为了便于和基站信息进行join此时需要将数据进行一次调整
//基站ID作为key 手机号和时长作为value
 val lacAndPT: RDD[(String, (String, Long))] = sumed.map(tup => {
val phone = tup._1._1
//用户手机号
val lac = tup._1._2 //基站ID
val time = tup._2 //用户在某个基站所停留的总时长
(lac, (phone, time))
})
//获取基站的基础数据
val lacInfo = sc.textFile("C:\\Users\\Administrator\\Desktop\\lacduration\\lac_info.txt")
//切分基站的书数据
val lacAndXY: RDD[(String, (String, String))] = lacInfo.map(line => {
val fields: Array[String] = line.split(",")
val lac = fields(0) //基站ID
val x = fields(1) // 经度
val y = fields(2) //纬度
(lac, (x, y))
})
//把经纬度的信息join到用户信息中
val joined: RDD[(String, ((String, Long), (String, String)))] = lacAndPT join lacAndXY
//为了方便以后的分组排序,需要进行数据整合
val phoneAndTXY: RDD[(String, Long, (String, String))] = joined.map(tup => {
val phone = tup._2._1._1 //手机号
val time = tup._2._1._2 //时长
val xy: (String, String) = tup._2._2 //经纬度
(phone, time, xy)
})
//按照用户的手机号进行分组
val grouped: RDD[(String, Iterable[(String, Long, (String, String))])] =
phoneAndTXY.groupBy(_._1)
//按照时长进行组内排序
val sorted: RDD[(String, List[(String, Long, (String, String))])] =
grouped.mapValues(_.toList.sortBy(_._2).reverse)
//数据进行整合
val res: RDD[(String, List[(Long, (String, String))])] = sorted.map(tup => {
val phone = tup._1
//手机号
val list = tup._2 //存储数据的集合
val filterList = list.map(tup1 => { //这个集合中的手机号顾虑掉
val time = tup1._2 //时长
val xy = tup1._3
(time, xy)
})
(phone, filterList)
})
 RDD的依赖关系
rdd之间有一些列的依赖关系,RDD和它依赖的父RDD的关系有两种即
宽依赖 和 窄依赖
窄依赖:父RDD和子RDD partitio之间的数据关系是一对一的或者父RDD的一个Partition只对应一个子RDD的
partition的情况下父RDD和子RDD partition的关系是多对一的(不会产生shuffle)
宽依赖;父RDD与子RDD partition之间的数据关系是一对多的关系, 会有shuffle的产生
宽窄依赖关系:
//取值
val ress = res.mapValues(_.take(2))
println(ress.collect.toList)
sc.stop()
}
}
```

## RDD的依赖关系 

rdd之间有一些列的依赖关系,RDD和它依赖的父RDD的关系有两种即
宽依赖 和 窄依赖
窄依赖:父RDD和子RDD partitio之间的数据关系是一对一的或者父RDD的一个Partition只对应一个子RDD的
partition的情况下父RDD和子RDD partition的关系是多对一的(不会产生shuffle)
宽依赖;父RDD与子RDD partition之间的数据关系是一对多的关系, 会有shuffle的产生
宽窄依赖关系: 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 001.png)

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 002.png)

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 003.png)

## Lineage(血统) 

RDD只支持粗粒度的转换,即在大量记录上执行的单个操作,将创建的RDD的一些列操作记录下来,以便恢复丢失的分区RDD的Lineage会记录RDD的元数据信息和转换型,若当前RDD的部分分区丢失,它会根据这个信息来重新运算和
恢复丢失的分区 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 005.png)

## 案例:基础案例 

```scala
package Day04
import java.net.URL
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
/*
用户点击产生日志信息,有时间戳和对应URL
URL中会有不用学科,统计学科的访问量
需求:根据用访问数据进行统计用户对各个学科的各个模块的访问量Top3
思路:1.统计每个模块的访问量
2. 按照学科进行分组
3. 学科进行排序
4. 取top3
*/
object SubjectDemo {
def main(args: Array[String]): Unit = {
val conf = new SparkConf().setAppName("SubjectDemo").setMaster("local")
val sc = new SparkContext(conf)
// 1.对数据进行切分
val tuples: RDD[(String, Int)] =
sc.textFile("C:\\Users\\Administrator\\Desktop\\subjectaccess\\access.txt").map(line => {
val fields: Array[String] = line.split("\t")
//取出url
val url = fields(1)
(url, 1)
})
//将相同url进行聚合,得到了各个学科的访问量RDD缓存方式
RDD可以用过 persist方法或cache方法将前面计算的结果进行一个缓存,但是并不是这两个方法被调用时立即缓存,
而是需要通过action算子触发才能进行缓存操作,该RDD将会被缓存到计算机的节点内存中,并提供重用
val sumed: RDD[(String, Int)] = tuples.reduceByKey(_+_)
//从url中获取学科的字段 数据组成式 学科, url 统计数量
val subjectAndUC: RDD[(String, String, Int)] = sumed.map(tup => {
val url = tup._1 //用户url
val count = tup._2 // 统计的访问数量
val subject = new URL(url).getHost //学科
(subject, url, count)
})
//按照学科信息进行分组
val grouped: RDD[(String, Iterable[(String, String, Int)])] = subjectAndUC.groupBy(_._1)
//对分组数据进行组内排序
val sorted: RDD[(String, List[(String, String, Int)])] =
grouped.mapValues(_.toList.sortBy(_._3).reverse)
//取top3
val res: RDD[(String, List[(String, String, Int)])] = sorted.mapValues(_.take(3))
println(res.collect.toList)
sc.stop()
} 
}
```

## RDD缓存方式 

RDD可以用过 persist方法或cache方法将前面计算的结果进行一个缓存,但是并不是这两个方法被调用时立即缓存,而是需要通过action算子触发才能进行缓存操作,该RDD将会被缓存到计算机的节点内存中,并提供重用 

```scala
/** Persist this RDD with the default storage level (`MEMORY_ONLY`). */
def persist(): this.type = persist(StorageLevel.MEMORY_ONLY)
/** Persist this RDD with the default storage level (`MEMORY_ONLY`). */
def cache(): this.type = persist()
存储时是有默认级别的 StorageLevel.MEMORY_ONLY (默认就是存到内存中)
useDisk: Boolean,useMemory: Boolean,useOffHeap: Boolean, deserialized:Boolean,replication: Int =1 
决定了下面参数的传入方式
是否是用磁盘 是否使用内存 是否使用堆外内存 是否反序列化 副本的个数
object StorageLevel {
val NONE = new StorageLevel(false, false, false, false)
val DISK_ONLY = new StorageLevel(true, false, false, false)
val DISK_ONLY_2 = new StorageLevel(true, false, false, false, 2)
val MEMORY_ONLY = new StorageLevel(false, true, false, true)
val MEMORY_ONLY_2 = new StorageLevel(false, true, false, true, 2)
val MEMORY_ONLY_SER = new StorageLevel(false, true, false, false)
val MEMORY_ONLY_SER_2 = new StorageLevel(false, true, false, false, 2)
val MEMORY_AND_DISK = new StorageLevel(true, true, false, true)val MEMORY_AND_DISK_2 = new StorageLevel(true, true, false, true, 2)
val MEMORY_AND_DISK_SER = new StorageLevel(true, true, false, false)
val MEMORY_AND_DISK_SER_2 = new StorageLevel(true, true, false, false, 2)
val OFF_HEAP = new StorageLevel(false, false, true, false)
} p
s:MEMORY_AND_DISK 先存储到内存中内存存储满了在存到磁盘
MEMORY_ONLY 内存只能存储内存大小的数据,超出的部分将不会再存储
因为内存中只存了一部分,少了一部分数据,这部分数据被加载时它会重新计算
堆外内存: 堆外内存是相对于对内内存而言,堆内内存是由JVM管理的,在平时java中创建对象都处于堆内内存,并且它是
遵守JVM的内存管理规则(GC垃圾回收机制),那么堆外内存就是存在于JVM管控之外的一块内存,它不受JVM的管控约束
缓存容易丢失,或者存储在内存的数据由于内存存储不足可能会被删掉.RDD的缓存容错机制保证了即缓存丢失也能保证正
确的的计算出内容,通过RDD的一些列转换,丢失的数据会被重算,由于RDD的各个Partition是独立存在,因此只需要计算
丢失部分的数据即可,并不需要计算全部的Partition
package Day04
/*
缓存
*/
import java.net.URL
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
object SubjectDemo2 {
def main(args: Array[String]): Unit = {
val conf = new SparkConf().setAppName("SubjectDemo").setMaster("local")
val sc = new SparkContext(conf)
//0.添加学科信息
val subjects = Array("http://java.learn.com","http://ui.learn.com",
"http://bigdata.learn.com","http://h5.learn.com","http://android.learn.com")
// 1.对数据进行切分
val tuples: RDD[(String, Int)] =
sc.textFile("C:\\Users\\Administrator\\Desktop\\subjectaccess\\access.txt").map(line => {
val fields: Array[String] = line.split("\t")
//取出url
val url = fields(1)
(url, 1)
})
//将相同url进行聚合,得到了各个学科的访问量
/*
缓存使用的场景:通常会将后期常用的数据进行缓存
特别是发生shuffle后的数据,因为shuffle过程的代价太大,所以经常在shuffle后进行缓存
cache默认是缓存到内存中,cache是transformation函数,所以需要一个action算子触发
*/
val sumed: RDD[(String, Int)] = tuples.reduceByKey(_+_).cache()
//因为学科信息已经存储到Array中
for(subject <- subjects){
//对学科信息进行过滤自定义分区
val filtered: RDD[(String, Int)] = sumed.filter(_._1.startsWith(subject))
val res: Array[(String, Int)] = filtered.sortBy(_._2,false).take(3)
println(res.toList)
} 
sc.stop()
}
}
```

## 自定义分区 

```scala
package Day04
import java.net.URL
import org.apache.spark.{HashPartitioner, Partitioner, SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import scala.collection.mutable
/**
* 自定分区
* 数据中有不同的学科,将输出的一个学科生成一个文件
*/
object SubjectDemo3 {
def main(args: Array[String]): Unit = {
val conf = new SparkConf().setAppName("SubjectDemo").setMaster("local")
val sc = new SparkContext(conf)
// 1.对数据进行切分
val tuples: RDD[(String, Int)] =
sc.textFile("C:\\Users\\Administrator\\Desktop\\subjectaccess\\access.txt").map(line => {
val fields: Array[String] = line.split("\t")
//取出url
val url = fields(1)
(url, 1)
})
//将相同url进行聚合,得到了各个学科的访问量
val sumed: RDD[(String, Int)] = tuples.reduceByKey(_+_).cache()
//从url中获取学科的字段 数据组成式 学科, url 统计数量
val subjectAndUC = sumed.map(tup => {
val url = tup._1 //用户url
val count = tup._2 // 统计的访问数量
val subject = new URL(url).getHost //学科
(subject, (url, count))
})
/*
在没有使用之定义分区器之前,可以使用系统所提供的HashPartitoner来进行分区操作
HashPartitoner是spark中非常重要的一个分区器,也是默认分区器,API中90%都是用的是它来进行的分区功能:依据RDD中key值的hashcode的值将数据取模后得到该key值对应的下一个RDD分区id值
支持key值为null的情况,当key为null的时候,返回的就是0
该分区器基本上适合所有RDD数据类型的数据进行分区操作
ps:由于Java中数据的hashcode是基于数组对象本身,而不是基于数组元素的内容,如果RDD的key是数据类型
那么可能导致内容一致的数据key没有办法分配到同一个RDD中,解决方案就是自定义分区器
在使用默认分区的时候,会出现数据倾斜的问题,最好的方式还是使用自定义分区
*/
// val per: RDD[(String, (String, Int))] = subjectAndUC.partitionBy(new HashPartitioner(3))
// per.saveAsTextFile("out1")
//将所有学科取出来
val subjects: Array[String] = subjectAndUC.keys.distinct.collect
//创建自定义分区器对象
val partitioner: SubjectPartitioner = new SubjectPartitioner(subjects)
//分区
val partitioned: RDD[(String, (String, Int))] = subjectAndUC.partitionBy(partitioner)
//取top3
val rs = partitioned.mapPartitions(it => {
val list = it.toList
val sorted = list.sortBy(_._2._2).reverse
val top3: List[(String, (String, Int))] = sorted.take(3)
//因为方法的返回值需要一个iterator
top3.iterator
})
//存储数据
rs.saveAsTextFile("out2")
sc.stop()
}
} /
**
* 自定义分区器需要继承Partitioner并实现对应方法
* @param subjects 学科数组
*/
class SubjectPartitioner(subjects:Array[String]) extends Partitioner{
//创建一个map集合用来存到分区号和学科
val subject = new mutable.HashMap[String,Int]()
//定义一个计数器,用来生成分区好
var i = 0
for(s <- subjects){
//存学科和分区
subject+=(s -> i)
i+=1 //分区自增
} /
/获取分区数
override def numPartitions: Int = subjects.sizeDAG
生成DAG叫做有向无环图,原始的RDD通过一系列的转换就形成了DAG,根据RDD之间的依赖关系的不同DAG的划分可以
得到不同Stage,对于窄依赖,partition的转换处理在Stage中完成计算,对于宽依赖,由于有shuffle存在,只能在父RDD
处理完后,才能开始接下来的计算,因此宽以来的划分需要使用到Stage
Spark会产生Shuffle的算子
//获取分区号(如果传入的key不存在,默认将数据存储到0分区)
override def getPartition(key: Any): Int = subject.getOrElse(key.toString,0)
}
```

## DAG生成 

DAG叫做有向无环图,原始的RDD通过一系列的转换就形成了DAG,根据RDD之间的依赖关系的不同DAG的划分可以得到不同Stage,对于窄依赖,partition的转换处理在Stage中完成计算,对于宽依赖,由于有shuffle存在,只能在父RDD
处理完后,才能开始接下来的计算,因此宽以来的划分需要使用到Stage 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 005.png)

Spark会产生Shuffle的算子 

```scala
去重 distinct
聚合 reduceBykey aggregateBykey combineByKey
分组 groupBy groupByKey
排序 sortByKey sortBy
重新分区 repartition coalesce
集合和表的操作 interasection subtract subtarctBykey join leftOuterJoin
```

# day5

## DAG生成 

DAG叫做有向无环图,原始的RDD通过一系列的转换就形成了DAG,根据RDD之间的依赖关系的不同DAG的划分可以得到不同Stage,对于窄依赖,partition的转换处理在Stage中完成计算,对于宽依赖,由于有shuffle存在,只能在父RDD处理完后,才能开始接下来的计算,因此宽以来的划分需要使用到Stage 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 009.png)

## 任务生成并提交的四个阶段 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 010.png)

## 在WebUI界面中查看Task 

1.可以通过自己画DAG来推算出程序中可以生成多少个task 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 011.png)

2.webUI界面查看task的生成 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 012.png)

因为在计算的时候textFile默认是2个partition, 整个计算流程是3个stage ,实际得到的task应该会是6个实际的到数量是 4个 	

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 013.png)

要是出现skipped那么就会减少对应的task,但是这是没有问题的并且是对任务出现skipped是正常,之所以出现是因为要计算的数据已经缓存在内存中,没有必要重复计算,所以出现skipped对结果是没有影响的,并且也是一种计算优化在发生shuffle过程中会发生shuffleWrite和ShuffleReadshuffle write:发生在shuffle之前,把要shuffle的数据写到磁盘上主要是为了保证数据的安全性和避免占用大量内存shuffle read:发生在shuffle之后,后面的RDD读取前面RDD的数据过程

查看当前DAG图 

## cache和checkpoint的使用时机 

cache在产生shuffle的时候使用cache来进行数缓存,因为shuffle消耗过大,为了避免这个过程出现问题或是shuffle之后数据丢失有中间结果数据或shuffl后的数据,最好是添加checkpoint 

checkpoint数据最好是存储到HDFS上
无论是cache还是checkpoint都是为了提高效率和保证数据安全性 

## Spark会产生Shuffle的算子 

```
去重 distinct
聚合 reduceBykey aggregateBykey combineByKey
分组 groupBy groupByKey
排序 sortByKey sortBy
重新分区 repartition coalesce
集合和表的操作 interasection subtract subtarctBykey join leftOuterJoin
```

## 案例:ip所属区域的访问量 

这些数据是用户访问所产生的日志信息http.log是用户访问网站所产生的日志,这些数据的产生是通过后台js埋点所得到的数据
数据组成基本上是:时间戳,IP地址, 访问网址, 访问数据 浏览器信息的等 ip.txt是 ip段数据 记录着一些ip段范围对应的位置
需求:通过http.log中的ip查找范围访问量那个多一些 

```scala
package Day05
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
/*
需求:根据用户访问的ip地址来统计所属区域并统计访问量
思路:1. 获取ip的基本数据
2. 把ip基础数据广播出去
3. 获取用户访问数据
4. 通过ip基础信息来判断用户数据那个区域
5. 通过得到的区域区域来统计访问量
6. 输出结果
*/
object IPSearch {
def main(args: Array[String]): Unit = {
//1.创建conf对象然后通过conf创建sc对象
val conf = new SparkConf().setAppName("ipsearch").setMaster("local")
val sc = new Spark Context(conf)
//2.获取ip的基础数据
val ipinfo = sc.textFile("C:\\Users\\Administrator\\Desktop\\ipsearch\\ip.txt")
//3.拆分数据
val splitedIP: RDD[(String, String, String)] = ipinfo.map(line => {
//数据是以|分割的
val fields = line.split("\\|")
//获取IP的起始和结束范围 取具体转换过后的值
val startIP = fields(2)
//起始
val endIP = fields(3) //结束
val shengfen = fields(6) // IP对应的省份
(startIP, endIP, shengfen)
})
/**
* 对于经常用到变量值,在分布式计算中,多个节点task一定会多次请求该变量
* 在请求过程中一定会产生大量的网络IO,因此会影响一定的计算性能
* 在这种情况下,可以使将该变量用于广播变量的方式广播到相对应的Executor端
* 以后再使用该变量时就可以直接冲本机获取该值计算即可,可以提高计算数据
*/
//在使用广播变量之前,需要将广播变量数据获取
val arrIPInfo: Array[(String, String, String)] = splitedIP.collect
//广播数据
val broadcastIPInfo: Broadcast[Array[(String, String, String)]] = sc.broadcast(arrIPInfo)
//获取用户数据
val userInfo = sc.textFile("C:\\Users\\Administrator\\Desktop\\ipsearch\\http.log")
//切分用户数据并查找该用户属于哪个省份
val shengfen: RDD[(String, Int)] = userInfo.map(line => {
//数据是以 | 分隔的
val fields = line.split("\\|")
//获取用户ip地址 125.125.124.2
val ip = fields(1)
val ipToLong = ip2Long(ip)
//获取广播变量中的数据
val arrInfo: Array[(String, String, String)] = broadcastIPInfo.value
//查找当前用ip的位置
//线性查找(遍历数据注意对比)
//二分查找(必须排序)
//i值的获取有两种形式:
//1.得到正确的下标,可以放心的去取值
//2.得到了-1 没有找到
//最好加一个判断若是 -1 写一句话 或是 直接结束
val i: Int = binarySearch(arrInfo, ipToLong)
val shengfen = arrInfo(i)._3
(shengfen, 1)
})
//统计区域访问量
val sumed = shengfen.reduceByKey(_+_)
//输出结果
println(sumed.collect.toList)
sc.stop()} /
**
* 把ip转换为long类型 直接给 125.125.124.2
* @param ip
* @return
*/
def ip2Long(ip: String): Long = {
val fragments: Array[String] = ip.split("[.]")
var ipNum = 0L
for (i <- 0 until fragments.length) {
//| 按位或 只要对应的二个二进位有一个为1时，结果位就为1 <<左位移
ipNum = fragments(i).toLong | ipNum << 8L
} 
ipNum
} /
**
* 通过二分查找来查询ip对应的索引
* *
/
def binarySearch(arr:Array[(String, String, String)],ip:Long):Int={
//开始和结束值
var start = 0
var end = arr.length-1
while(start <= end){
//求中间值
val middle = (start+end)/2
//arr(middle)获取数据中的元组\
//元组存储着ip开始 ip结束 省份
//因为需要判断时候在ip的范围之内.,所以需要取出元组中的值
//若这个条件满足就说明已经找到了ip
if((ip >= arr(middle)._1.toLong) &&(ip<=arr(middle)._2.toLong)){
return middle
}
 if(ip < arr(middle)._1.toLong){
end = middle -1
}else{
start = middle+1
}
} -1
}
}
```

将数据存储到了Mysql数据库,需要在pom.xml中添加依赖 

```
<dependency>
<groupId>mysql</groupId>
<artifactId>mysql-connector-java</artifactId>
<version>5.1.38</version>
</dependency>
```

```scala
package Day05
import java.sql.{Connection, Date, DriverManager, PreparedStatement}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
//将数据固话到数据库中
object IPSearchJDBC {
def main(args: Array[String]): Unit = {
//1.创建conf对象然后通过conf创建sc对象
val conf = new SparkConf().setAppName("ipsearch").setMaster("local")
val sc = new SparkContext(conf)
//2.获取ip的基础数据
val ipinfo = sc.textFile("C:\\Users\\Administrator\\Desktop\\ipsearch\\ip.txt")
//3.拆分数据
val splitedIP: RDD[(String, String, String)] = ipinfo.map(line => {
//数据是以|分割的
val fields = line.split("\\|")
//获取IP的起始和结束范围 取具体转换过后的值
val startIP = fields(2)
//起始
val endIP = fields(3) //结束
val shengfen = fields(6) // IP对应的省份
(startIP, endIP, shengfen)
})
/**
* 对于经常用到变量值,在分布式计算中,多个节点task一定会多次请求该变量
* 在请求过程中一定会产生大量的网络IO,因此会影响一定的计算性能
* 在这种情况下,可以使将该变量用于广播变量的方式广播到相对应的Executor端
* 以后再使用该变量时就可以直接冲本机获取该值计算即可,可以提高计算数据
*/
//在使用广播变量之前,需要将广播变量数据获取val arrIPInfo: Array[(String, String, String)] = splitedIP.collect
//广播数据
val broadcastIPInfo: Broadcast[Array[(String, String, String)]] = sc.broadcast(arrIPInfo)
//获取用户数据
val userInfo = sc.textFile("C:\\Users\\Administrator\\Desktop\\ipsearch\\http.log")
//切分用户数据并查找该用户属于哪个省份
val shengfen: RDD[(String, Int)] = userInfo.map(line => {
//数据是以 | 分隔的
val fields = line.split("\\|")
//获取用户ip地址 125.125.124.2
val ip = fields(1)
val ipToLong = ip2Long(ip)
//获取广播变量中的数据
val arrInfo: Array[(String, String, String)] = broadcastIPInfo.value
//查找当前用ip的位置
//线性查找(遍历数据注意对比)
//二分查找(必须排序)
//i值的获取有两种形式:
//1.得到正确的下标,可以放心的去取值
//2.得到了-1 没有找到
//最好加一个判断若是 -1 写一句话 或是 直接结束
val i: Int = binarySearch(arrInfo, ipToLong)
val shengfen = arrInfo(i)._3
(shengfen, 1)
})
//统计区域访问量
val sumed = shengfen.reduceByKey(_+_)
//算子可以用来对数据库存储数据
sumed.foreachPartition(data2MySql)
println("数据库存储成功")
//输出结果
println(sumed.collect.toList)
sc.stop()
} /
**
* 用于发结果存储到MySQL数据中的函数* *
/
val data2MySql = (it:Iterator[(String, Int)])=>{
//创建连接对象和预编译语句
var conn:Connection = null
//已经包含了编译好的SQL语句,并且可以使用 ?占位
var ps:PreparedStatement = null
///插入的sql语句
val sql = "insert into location_info(location,counts,access_date)values(?,?,?)"
//jdbc连接驱动设置
val jdbcurl = "jdbc:mysql://10.0.15.20:3306/mydb1?useUnicode=true&characterEncoding=utf8"
val user = "root"
val password = "123456"
try {
//获取连接
conn = DriverManager.getConnection(jdbcurl, user, password)
//向语句中添加数据
it.foreach(line => {
ps = conn.prepareStatement(sql)
//?的顺序是从1开始逐渐递增
ps.setString(1, line._1)
ps.setInt(2, line._2)
ps.setDate(3, new Date(System.currentTimeMillis()))
ps.executeUpdate()
})
}catch{
case e:Exception => println(e.printStackTrace())
}finally{
if(ps!=null){
ps.close()
} i
f(conn!=null){
conn.close()
}
}
}
/**
* 把ip转换为long类型 直接给 125.125.124.2
* @param ip
* @return
*/
def ip2Long(ip: String): Long = {
val fragments: Array[String] = ip.split("[.]")
var ipNum = 0L
for (i <- 0 until fragments.length) {
//| 按位或 只要对应的二个二进位有一个为1时，结果位就为1 <<左位移
ipNum = fragments(i).toLong | ipNum << 8L
}ps:mysql远程连接时出现连接问题
Host *(IP) is not allowed to connect to this MySQL Server
1.登录mysql
2.执行 use mysql
3.
ipNum
} /
**
* 通过二分查找来查询ip对应的索引
* *
/
def binarySearch(arr:Array[(String, String, String)],ip:Long):Int={
//开始和结束值
var start = 0
var end = arr.length-1
while(start <= end){
//求中间值
val middle = (start+end)/2
//arr(middle)获取数据中的元组\
//元组存储着ip开始 ip结束 省份
//因为需要判断时候在ip的范围之内.,所以需要取出元组中的值
//若这个条件满足就说明已经找到了ip
if((ip >= arr(middle)._1.toLong) &&(ip<=arr(middle)._2.toLong)){
return middle
} i
f(ip < arr(middle)._1.toLong){
end = middle -1
}else{
start = middle+1
}
} -1
}
}
```

ps:mysql远程连接时出现连接问题
Host *(IP) is not allowed to connect to this MySQL Server
1.登录mysql
2.执行 use mysql
3.是localhost就需要更改权限,更改成% --> 所有 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 014.png)

4.update user set host = '%' where user = 'root'
ps:偶尔会出现报错,无视
5.FLUSH PRIVILEGES 刷新权限 

## JDBCRDD 

spark提供了一个RDD来处理对JDBC的连接,但是十分的鸡肋.这个RDD只能进行查询,不能进行增删改,很少用 

```scala
package Day05
import java.sql.{Date, DriverManager}
import org.apache.spark.rdd.JdbcRDD
import org.apache.spark.{SparkConf, SparkContext}
//spark提供JDBCRDD
object JDBCRDDTest {
def main(args: Array[String]): Unit = {
val conf = new SparkConf().setAppName("JDBCRDDTest").setMaster("local")
val sc = new SparkContext(conf)
///插入的sql语句
val sql = "select id,location,counts,access_date from location_info where id >= ? and id <=
?"
//jdbc连接驱动设置
val jdbcurl = "jdbc:mysql://10.0.15.20:3306/mydb1?useUnicode=true&characterEncoding=utf8"
val user = "root"
val password = "123456"
//获取连接
val conn =()=>{
Class.forName("com.mysql.jdbc.Driver").newInstance()
DriverManager.getConnection(jdbcurl,user,password)
}
/*
1.获取哦sparkContext对象
2. Connection连接对象
3.查询sql语句
4和5是获取数据的范围
6. partition个数
7.最终结果的返回
*/
val jdbcRDD: JdbcRDD[(Int, String, Int, Date)] = new JdbcRDD(
sc,
conn,
sql,
0,
200,Accumulator累加器
累加器是Saprk提供,用于多个task并发的对某个变量进行操作,task可以对累加器进行操作,不能读取其值,只有在
Driver才能读取
ps:可以看做静态全局变量
1,
res => {
//res查询之后获得的结果
//通过get方法的重载形式传入不同 类名 得到数据
val id = res.getInt("id")
val location = res.getString("location")
val counts = res.getInt("counts")
val access_date = res.getDate("access_date")
//JdbcRDD最终要返回这个查询的结果
(id, location, counts, access_date)
}
) 
println( jdbcRDD.collect.toList)
sc.stop()
}
}
```

## Accumulator累加器 

累加器是Saprk提供,用于多个task并发的对某个变量进行操作,task可以对累加器进行操作,不能读取其值,只有在Driver才能读取
ps:可以看做静态全局变量 

```scala
package Day05
import org.apache.spark.{SparkConf, SparkContext}
//演示累加器
object AccumulatorDemo {
def main(args: Array[String]): Unit = {
val conf = new SparkConf().setAppName("JDBCRDDTest").setMaster("local[2]")
val sc = new SparkContext(conf)
val numbers = sc .parallelize(List(1,2,3,4,5,6),2)
println(numbers.partitions.length)
val sum = sc.accumulator(0)
//为什么sum值通过计算过后还是0
//因为foreach是没有返回值,整个计算过程都是在executor端完后
//foreach是在driver端运行所以打印的就是 0,foreach没有办法获取数据
//var sum = 0
numbers.foreach(num =>{
println("这个是foreach中的num:"+num)
sum+= num
println("这个是foreach中的sum:"+sum)
})
println(sum)
sc.stop()作用:
1.能够精确的统计数据的各种数据例如:可以统计出符合userID的记录数
在同一个时间段内产生了多少次购买,可以使用ETL进行数据清洗,并使用Accumulator来进行数据的统计
2.作为调试工具,能够观察每个task的信息,通过累加器可以在sparkIUI观察到每个task所处理的记录数
自定义排序
比如有一个自定义类型,类型中有很多字段,现在要对某些字段进行排序
}
}
```

作用:
1.能够精确的统计数据的各种数据例如:可以统计出符合userID的记录数
在同一个时间段内产生了多少次购买,可以使用ETL进行数据清洗,并使用Accumulator来进行数据的统计
2.作为调试工具,能够观察每个task的信息,通过累加器可以在sparkIUI观察到每个task所处理的记录数 

## 自定义排序 

比如有一个自定义类型,类型中有很多字段,现在要对某些字段进行排序 

```scala
package Day05
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
//自定义排序
object CustomSortTest {
def main(args: Array[String]): Unit = {
val conf = new SparkConf().setAppName("JDBCRDDTest").setMaster("local")
val sc = new SparkContext(conf)
val girlInfo = sc.parallelize(List(("xiaoming",90,32),("xiaohong",70,32),("xiaobai",80,34),
("bobo",90,35)))
//根据颜值排序(简单)
val sorted: RDD[(String, Int, Int)] = girlInfo.sortBy(_._2,false)
println(sorted.collect.toList)
//第一种自定义排序
import MyOrdering.girlOrdering
val sorted1: RDD[(String, Int, Int)] = girlInfo.sortBy(g => Girl(g._2,g._3))
println(sorted1.collect.toList)
//第二种排序方式自定义排序
val sorted2: RDD[(String, Int, Int)] = girlInfo.sortBy(g => Girls(g._2,g._3))
println(sorted2.collect.toList)
} /
/样例类来存储数据
case class Girl(faceValue:Int,age:Int)
//第二个种排序
case class Girls(faceValue:Int,age:Int) extends Ordered[Girls]{
override def compare(that: Girls): Int = {
if(this.faceValue == that.faceValue){
that.age - this.age
}else{
this.faceValue - that.faceValue
    
}
}
}
} //sparkcore问题:
/隐式转换类型
//若使用隐式转换函数必须使用单例类
object MyOrdering {
//进行隐式转换 自定义比较(隐式转换函数)
implicit val girlOrdering = new Ordering[Girl]{
override def compare(x: Girl, y: Girl): Int = {
if(x.faceValue != y.faceValue){
x.faceValue - y.faceValue
}else{
y.age - x.age
}
}
}
}
```

## sparkcore问题: 

```
1.sparkcontext是在那一端生成的? driver
2.RDD是在那一端生成的? driver
3.调用RDD算子是在哪一端调用? driver
4.RDD在调用当前算子的时候,算子需要传入一个函数,函数的声明和传入那一端 driver
5.RDD在调用当前算子的时候,涮粗需要传入函数,函数的是在那一端执行的业务逻辑 worker(executor)
6.自定义分区器这个类是在哪一端被实例化的 driver
7.分区器器中的getPartition方法时在那一端调用 worker(executor)
8.DAG是在那一端构建的 driver
9.DAG是在哪一端被划分成多个stage的 driver
10.DAG是那个类完成切分stage的功能 DAGSheduler(特质)DAGShedulerImpl
11.DAGSheduler将切分好的stage以什么样式传递给TaskSheduler -->TaskSet
12.Task生成在那一端 driver
13.广播变量是在哪一端调用方法进行广播的 driver
14.要广播变量数据应该在哪一端创建好在广播 driver
```

# day6

## 什么是SparkSQL 

spark1.0版本就已经推出SaprkSQL最早叫shark
Shark是基于spark框架并且兼容hive执行SQL执行引擎,因为底层使用了Spark,比MR的Hive普遍的要快上2倍左右,
当数据全部load内存中,此时会比Hive块10倍以上,SparkSQL就是一种交互式查询应用服务
特点:
1.内存列存储--可以大大优化内存的使用率,减少内存消耗,避免GC对大量数据性能的开销
2.字节码生成技术-- 可以使用动态的字节码技术优化性能
3.Scala代码的优化
SparkSql官网就是spark:http://spark.apache.org/sql/ 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 015.png)

SaprkSql是spark用来处理结构化的一个模块,它提供一个抽象的数据集DateFrame,并且是作为分布式SQL查询引擎的应用
为什么要学习sparkSql
之前已经学习了hive,它将HiveSql转换成MR然后提交到集群上执行,减少编写MR程序的复杂性,但是因为采用即计算框架是MR所以执行效率比较慢,SparkSql就应运而生. 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 016.png)

1.易整合2.统一的数据访问方式,3兼容hive,4提供了统一的数据连接方式(JDBC/ODBC) 

## DataFrames 

与RDD类型,DataFrame也是一个分布式数据容器,然而DataFrame更像传统数据库中二维表格,除了记录数据之外,还记数据的结构信息(schema),同时与Hive类型,DataFrame也支持嵌套数据类型(sturct,map和array),从API易用的角度来看,DataFrame提供更高级的API,比函数RDDAPI更加友好 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 017.png)

## 创建DataFrame 

saprk-shell版本
spark-shell中已经创建好了SparkContext和SQLContext对象 

```scala
//创建了一个数据集,实现了并行化
val seq1 = Seq(("1","xiaoming",15),("2","xiaohong",20),("3","xiaobai",10))
val rdd1 = sc.parallelize(seq1)
//将当前rdd对象转换成DataFrame对象(数据信息和数据结构信息存储到了DataFrame)
//_1: string, _2: string, _3: int
//在使用toDF进行转换的时候,空参的情况下,默认是 _+数字 作为列名 数字从1开始逐渐递增
val df = rdd1.toDF
//查看数据 show 算子来打印 show是一个action类型 算子
df.show
DSL风格语法
领域特定语言（英语：domain-specific language、DSL）指的是专注于某个应用程序领域的计算机语言。又译作领域
专用语言。
查询:
df.select("name").show
df.select("name","age").show
//条件过滤filter
df.select("name","age").filter("age>10").show
ps:参数必须是一个字符串 filter中的表达也需要是一个字符串
//参数是类名col("列名")
df.select("name","age").filter(col("age")>10).show
//分组统计个数
df.groupBy("age").count().show()
//打印dataFrame结构信息
df.printSchema
SQL风格语法
1.将DataFrame注册成表(临时表),表会被存储到Sqlcontext中以编码的形式来执行sparkSQL
先将工程中的maven的pom.xml文件添加依赖
第一种方式通过反射推断
df.registerTempTable("t_person")
查询语法:需要通过sqlContext对象调用sql方法写入sql语句
sqlContext.sql("select name,age from t_person where age > 10").show
sqlContext.sql("select * from t_person order by age desc limit 2").show
hive中orderby和sortby的区别?
使用orderby全局排序
使用distribute和sort进行分组排序
distribute by+sort by 通过distribute by 设定字段为key,数据会被hash到不同reduce机器上
然后同sort by会对同一个reduce机器上的数据进行局部排序
orderby是全局有序 distribute+sort 局部有序全局无序
结构表信息:
sqlContext.sql("desc t_person").show
```

## 以编码的形式来执行sparkSQL 

先将工程中的maven的pom.xml文件添加依赖 

```xml
<dependency>
<groupId>org.apache.spark</groupId>
<artifactId>spark-sql_2.10</artifactId>
<version>1.6.3</version>
</dependency>
```

## 第一种方式通过反射推断 

```scala
package Day06
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
//通过反射的形式来获取
object SparkSQLDemo1 {
def main(args: Array[String]): Unit = {
//之前在spark-shell中 sparkConetxt和SqlContext是创建好的所以不需要创建直接使用
//因为是代码编程,需要进行创建
val conf = new SparkConf().setAppName("SparkSQLDemo1").setMaster("local")
val sc = new SparkContext(conf)
//创建SQLContext对象
val sqlc = new SQLContext(sc)
//集群中获取数据生成RDD
val lineRDD: RDD[Array[String]] =
sc.textFile("hdfs://hadoop01:8020/Person.txt").map(_.split(" "))
// lineRDD.foreach(x=>println(x.toList))//将获取的数据关联到样例类中
val personRDD: RDD[Person] = lineRDD.map(x => Person(x(0).toInt,x(1),x(2).toInt))
import sqlc.implicits._
//toDF相当于是反射,这里需要使用的话需要添加导入包import sqlc.implicits._
/*
DataFrame[_1:Int,_2:String,_3:Int]
saprk-shell 数据是一个自己生成并行化数据并没有使用样例类来存数据而是直接使用
直接调用toDF的时候,使用就说默认列名 _ + 数字 数字从1开始逐渐递增
可以在调用toDF方法时候指定类的名称(指定名称多余数据会报错)
数据要和列名一一对象
使用代码编程数据是存储到样例类中,样例类中的构造方法中的参数就是对应的列名
所以通过toDF可以直接获取到对应的属性名作为列名使用
同时也可以指定自定义列名.
*/
val personDF: DataFrame = personRDD.toDF
personDF.show()
//使用Sql语法
//注册临时表 ,这个表相当于存储在了 SQLContext所创建对象中
personDF.registerTempTable("t_person")
val sql = "select * from t_person where age > 20 order by age"
//查询
val res = sqlc.sql(sql)
//默认打印是20行,可以自定义打印行数
res.show()
//固化数据
//将数据写到文件中mode是以什么形式写 写成什么文件
/**
* "overwrite" 复写 "append" 追加
*/
// res.write.mode("append").json("out3")
//hdfs://hadoop01:8020/output7 写到hdfs上
//除了这两种还可以cvs模式 , jdbc模式
//cvs在1.6.3spark中需要集成第三方插件,才能使用,2.0之后自动集成
//这个方法不要使用因为在2.0会被删除
//res.write.mode("append").save("hdfs://hadoop01:8020/output7")
sc.stop()
}
    case class Person(id:Int,name:String,age:Int)
} 将
当前成进行一个打包操作提交到集群,需要做一定的更改,注意path路径 修改成 args(下标)
spark-submit
--class 类名(类的全限定名(包名+类名))
--master spark://集群名:7077通过StructType
/root/jar包路径
输入数据路径
输出数据路径
查看运行结果(多个文件的情况下)
hdfs dfs -cat /输入文件路径/part-r-*
```

## 通过StructType 

```scala
package Day06
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.{SparkConf, SparkContext}
object SparkSQLStructTypeDemo {
def main(args: Array[String]): Unit = {
val conf = new SparkConf().setAppName("SparkSQLStructTypeDemo").setMaster("local")
val sc = new SparkContext(conf)
val sqlcontext = new SQLContext(sc)
// 获取数据并拆分
val lineRDD = sc.textFile("hdfs://hadoop01:8020/Person.txt").map(_.split(" "))
//创建StructType对象 封装了数据结构(类似于表的结构)
val structType: StructType = StructType {
List(
// 列名 数据类型 是否可以为空值
StructField("id", IntegerType, false),
StructField("name", StringType, true),
StructField("age", IntegerType, true)
//列需要和数据对应,但是StructType这种可以列的数量大于数据,所有列对应的值应该是null
//列数是不能小于数据,不然会抛异常
// StructField("oop", IntegerType, false),
// StructField("poo", IntegerType, true)
)
} /
/val s = StructType(Array(StructField("id", IntegerType, false),StructField("name",
StringType, true),StructField("age", IntegerType, true)))
//将数据进行一个映射操作
val rowRDD: RDD[Row] = lineRDD.map(arr => Row(arr(0).toInt,arr(1),arr(2).toInt))
//将RDD转换为DataFrame
val personDF: DataFrame = sqlcontext.createDataFrame(rowRDD,structType)
personDF.show
//查询 存储都和SparkSQLDemo1中相同JDBC数据源
SparkSQL可以通过JBDC从关系型数据库中读取数据的方式创建DataFrame,在通过对DataFrame的一系列操作,还
可以将数据写会到关系型数据库中
使用spark-shell
必须执行mysql的连接驱动jar
将数据写入到Mysql中
}
}
```

## JDBC数据源 

SparkSQL可以通过JBDC从关系型数据库中读取数据的方式创建DataFrame,在通过对DataFrame的一系列操作,还可以将数据写会到关系型数据库中
使用spark-shell
必须执行mysql的连接驱动jar 

```scala
spark-shell \
--master spark://hadoop01:7077 \
--executor-memory 512m \
--total-executor-cores 2 \
--jars /root/mysql-connector-java-5.1.32.jar \
--driver-class-path /root/mysql-connector-java-5.1.32.jar
从mysql中加载数据生成DataFrame
//jdbcDF 就是DataFrame类型
val jdbcDF = sqlContext.read.format("jdbc").options(Map("url"->"jdbc:mysql://10
.0.15.20:3306/mydb1","driver"->"com.mysql.jdbc.Driver","dbtable"->"location_info","user"-
>"root","password"->"123456")).load()
jdbcDF.show
```

将数据写入到Mysql中 

```scala
package Day06
import java.util.Properties
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.{SparkConf, SparkContext}
object DataFormeInputJDBC {
def main(args: Array[String]): Unit = {
val conf = new SparkConf().setAppName("DataFormeInputJDBC").setMaster("local")
val sc = new SparkContext(conf)
val sqlcontext = new SQLContext(sc)
//获取数据拆分
val liens = sc.textFile("hdfs://hadoop01:8020/Person.txt").map(_.split(" "))
Hive-on-Sparkhive底层是通MR进行计算,将其改变为SparkCore来执行
配置步骤
1.在不适用高可用集群的前提下,只需要将Hadoop安装目录中的core-site.xml拷贝到spark的配置conf文件目录下
即可
2.将Hive安装路下的hive-site.xml拷贝到spark的配置conf文件目录下即可
若是高可用:需要将Hadoop安装路径下的core-site.xml和hdfs-site.xml拷贝到saprk conf目录下
操作完成之后建议重启集群
通过sparksql来操作,需要在spark安装路径中bin目录下启动
//StructType 存的表结构
val structType = StructType(Array(StructField("id",IntegerType,false),
StructField("name",StringType,true),StructField("age",IntegerType,true)))
//开始映射
val rowRDD: RDD[Row] = liens.map(arr => Row(arr(0).toInt,arr(1),arr(2).toInt))
//将当前RDD 转换为 DataFrame
val personDF: DataFrame = sqlcontext.createDataFrame(rowRDD,structType)
//创建一个用于写入mysql配置信息
val prop = new Properties()
prop.put("user","root")
prop.put("password","123456")
prop.put("driver","com.mysql.jdbc.Driver")
//提供mysql的URL
val jdbcurl = "jdbc:mysql://10.0.15.20:3306/mydb1"
//表名
val table = "person"
//数据库要对,表若不存在会自动创建并存储数据
//需要将数据写入到jdbc
personDF.write.mode("append").jdbc(jdbcurl,table,prop)
println("插入数据成功")
sc.stop()
}
}
```

Hive-on-Spark
hive底层是通MR进行计算,将其改变为SparkCore来执行
配置步骤
1.在不适用高可用集群的前提下,只需要将Hadoop安装目录中的core-site.xml拷贝到spark的配置conf文件目录下
即可
2.将Hive安装路下的hive-site.xml拷贝到spark的配置conf文件目录下即可
若是高可用:需要将Hadoop安装路径下的core-site.xml和hdfs-site.xml拷贝到saprk conf目录下
操作完成之后建议重启集群
通过sparksql来操作,需要在spark安装路径中bin目录下启动 

```scala
启动:
spark-sql \
--master spark://hadoop01:7077 \
--executor-memory 512m \
--total-executor-cores 2 \
--jars /root/mysql-connector-java-5.1.32.jar \
--driver-class-path /root/mysql-connector-java-5.1.32.jar
基本操作:
创建表:
create table person(id int,name string,age int)row format delimited fields terminated by ' ';
加载数据(本地加载)load data local inpath "/root/Person.txt" into table person;
查询:
select* from person;
select name,age from person where age > 20 order by age;
删除
drop table person;
内部表 和 外部表
表没有被 external修饰的都是内部表 ,被修饰就是外部表
hive本事不能存储数据,依托于HDFS
内部表存储数据被删除,同时会删除数据和源信息
外部表存储数据被删除,仅仅会删除源数据, HDFS中存储的数据会被保留下来
```

# day7

## 什么是kafka 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 021.png)

**kafka是一个开源的项目是Apache基金会,有scala编写,该设计目标是为了处理实时数据提供一个统一的高吞吐量,低等待的一个平台Kafka是一个分布式消息队列,生产者和消费者的功能,它提供了类似于JMS的特性,但是在设计实现上完全不相同,此外它并不是JMS规范的实现** 

kaka对消息的保存根据Topic进行归类,发送消息称为Producer,接收消息的称为Consumer
此外Kafka集群有多个kafka实例组成,每个实例(server)称为broker
这些都依赖于Zookeeper保存一些meta信息,用来保证系统的可用性
ps:kafka是不提供生产者和消费者这只是一个概念,并且kafa的数据模型所提供
例如 flume从logServer中拉取数据到kafka,flume就是相遇生产者
SparkStreaming从kafka中获取数据,SparkStreaming就相当于消费者
官网:http://kafka.apache.org/ 

## 什么JMS（了解） 

**JMS的基础** 

JMS是什么：JMS是Java提供的一套技术规范
JMS干什么用：用来异构系统 集成通信，缓解系统瓶颈，提高系统的伸缩性增强系统用户体验，使得系统模块化和
组件化变得可行并更加灵活
通过什么方式：生产消费者模式（生产者、服务器、消费者） 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 022.png)

## 2.2、JMS消息传输模型 

l 点对点模式（一对一，消费者主动**拉取数据，消息收到后消息清除）**
点对点模型通常是一个基于拉取或者轮询的消息传送模型，这种模型从队列中请求信息，而不是将消息推送到客户
端。这个模型的特点是发送到队列的消息被一个且只有一个接收者接收处理，即使有多个消息监听者也是如此。
l 发布/订阅模式（一对多，数据生产后，**推送给所有订阅者）**
发布订阅模型则是一个基于推送的消息传送模型。发布订阅模型可以有多种不同的订阅者，临时订阅者只在主动监
听主题时才接收消息，而持久订阅者则监听主题的所有消息，即当前订阅者不可用，处于离线状态。 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 023.png)

queue.put（object） 数据生产
queue.take(object) 数据消费 

## 2.3、JMS核心组件 

l Destination：消息发送的目的地，也就是前面说的Queue和Topic。
l Message ：从字面上就可以看出是被发送的消息。
l Producer： 消息的生产者，要发送一个消息，必须通过这个生产者来发送。
l MessageConsumer： 与生产者相对应，这是消息的消费者或接收者，通过它来接收一个消息。 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 024.png)

通过与ConnectionFactory可以获得一个connection
通过connection可以获得一个session会话。 

## 2.4、常见的类JMS消息服务器 

### 2.4.1、JMS消息服务器 ActiveMQ 

ActiveMQ 是Apache出品，最流行的，能力强劲的开源消息总线。ActiveMQ 是一个完全支持JMS1.1和J2EE 1.4规
范的。
主要特点：
l 多种语言和协议编写客户端。语言: Java, C, C++, C#, Ruby, Perl, Python, PHP。应用协议: OpenWire,Stomp
REST,WS Notification,XMPP,AMQP
l 完全支持JMS1.1和J2EE 1.4规范 (持久化,XA消息,事务)
l 对Spring的支持,ActiveMQ可以很容易内嵌到使用Spring的系统里面去,而且也支持Spring2.0的特性
l 通过了常见J2EE服务器(如 Geronimo,JBoss 4, GlassFish,WebLogic)的测试,其中通过JCA 1.5 resource adaptors
的配置,可以让ActiveMQ可以自动的部署到任何兼容J2EE 1.4 商业服务器上
l 支持多种传送协议:in-VM,TCP,SSL,NIO,UDP,JGroups,JXTA
l 支持通过JDBC和journal提供高速的消息持久化
l 从设计上保证了高性能的集群,客户端-服务器,点对点
l 支持Ajax
l 支持与Axis的整合
l 可以很容易得调用内嵌JMS provider,进行测试 

### 2.4.2、分布式消息中间件 Metamorphosis 

Metamorphosis (MetaQ) 是一个高性能、高可用、可扩展的分布式消息中间件，类似于LinkedIn的Kafka，具有消
息存储顺序写、吞吐量大和支持本地和XA事务等特性，适用于大吞吐量、顺序消息、广播和日志数据传输等场景，
在淘宝和支付宝有着广泛的应用，现已开源。
主要特点：
l 生产者、服务器和消费者都可分布
l 消息存储顺序写
l 性能极高,吞吐量大
l 支持消息顺序
l 支持本地和XA事务
l 客户端pull，随机读,利用sendfile系统调用，zero-copy ,批量拉数据
l 支持消费端事务
l 支持消息广播模式
l 支持异步发送消息
l 支持http协议
l 支持消息重试和recover
l 数据迁移、扩容对用户透明
l 消费状态保存在客户端
l 支持同步和异步复制两种HA
l 支持group commit

### 2.4.3、分布式消息中间件 RocketMQ

RocketMQ 是一款分布式、队列模型的消息中间件，具有以下特点：
l 能够保证严格的消息顺序
l 提供丰富的消息拉取模式
l 高效的订阅者水平扩展能力
l 实时的消息订阅机制
l 亿级消息堆积能力
l Metaq3.0 版本改名，产品名称改为RocketMQ
2.4.4、其他MQ
l .NET消息中间件 DotNetMQ
l 基于HBase的消息队列 HQueue
l Go 的 MQ 框架 KiteQ
l AMQP消息服务器 RabbitMQ
l MemcacheQ 是一个基于 MemcacheDB 的消息队列服务器 

## 3、为什么需要消息队列（重要、了解） 

消息系统的核心作用就是三点：解耦，异步和并行
以用户注册的案列来说明消息系统的作用

### 3.1、用户注册的一般流程 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 025.png)

问题：随着后端流程越来越多，每步流程都需要额外的耗费很多时间，从而会导致用户更长的等待延迟。 

### 3.2、用户注册的并行执行 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 026.png)

问题：系统并行的发起了4个请求，4个请求中，如果某一个环节执行1分钟，其他环节再快，用户也需要等待1分钟。如果其中一个环节异常之后，整个服务挂掉了 

### 3.3、用户注册的最终一致 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 027.png)

1、 保证主流程的正常执行、执行成功之后，发送MQ消息出去。
2、 需要这个destination的其他系统通过消费数据再执行，最终一致。 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 028.png)

## Kafka中重要的角色 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 029.png)

**Kafka名词解释和工作方式**
Producer ：消息生产者，就是向kafka broker发消息的客户端。
Consumer ：消息消费者，向kafka broker拉取消息的客户端
Topic ：我们可以理解为一个队列。
Consumer Group （CG）：这是kafka用来实现一个topic消息的广播（发给所有的consumer）和单播（发给任意
一个consumer）的手段。一个topic可以有多个CG。topic的消息会复制（不是真的复制，是概念上的）到所有的
CG，但每个partion只会把消息发给该CG中的一个consumer。如果需要实现广播，只要每个consumer有一个独
立的CG就可以了。要实现单播只要所有的consumer在同一个CG。用CG还可以将consumer进行自由的分组而不
需要多次发送消息到不同的topic。
Broker ：一台kafka服务器就是一个broker。一个集群由多个broker组成。一个broker可以容纳多个topic。
Partition：为了实现扩展性，一个非常大的topic可以分布到多个broker（即服务器）上，一个topic可以分为多个
partition，每个partition是一个有序的队列。partition中的每条消息都会被分配一个有序的id（offset）。kafka只
保证按一个partition中的顺序将消息发给consumer，不保证一个topic的整体（多个partition间）的顺序。
Offset：kafka的存储文件都是按照offset.kafka来命名，用offset做名字的好处是方便查找。例如你想找位于2049
的位置，只要找到2048.kafka的文件即可。当然the first offset就是00000000000.kafka 

## Kafka常见问题: 

```
1.Segment的概念?
tpoic中会有一个到多个分区,分个分区中会有多个segment
segment的大小是在kafka文件中进行配置的
segment的大小是相等,每个segment有多个index文件和数据文件是一一对应的
ps:index文件中存储的是所以也就是文件存储的位置
2.数据存储机制?
首先是broker接收到数据,将数据放到操作系统中(Linux)的缓存中(pagecache)
pagecache会尽可能的使用空闲内存
会使用sendfile技术尽可能减少操作系统和应用程序之间重复缓存
写入数据的时候还会用到顺序写入的方式Consumer的负载均衡
当一个group中,有consumer加入或者离开时,会触发partitions负载均衡.负载均衡的最终目的,是提升topic的并发消
费能力，步骤如下：
1、 假如topic1,具有如下partitions: P0,P1,P2,P3
2、 加入group中,有如下consumer: C1,C2
3、 首先根据partition索引号对partitions排序: P0,P1,P2,P3
4、 根据consumer.id排序: C1,C2
5、 计算倍数: M = [P0,P1,P2,P3].size / [C1,C2].size,本例值M=2(向上取整)
6、 然后依次分配partitions: C1 = [P0,P3],C2=[P1,P2],即Ci = [P(i * M),P((i + 1) * M -1)]
写书数据的数据可以到600m/s(300~400m/s) 读取 20m/s ~30m/s
3.consumer是怎么解决负载均衡
当同一个组的consumer的数量发生改变的时候,触发kafka的负载均衡
首先会获取consumer消费的起始分区号,在计算出consumer要消费的分区数量
最后用起始分区号的hashcode值取余分区数
```

## Consumer的负载均衡 

当一个group中,有consumer加入或者离开时,会触发partitions负载均衡.负载均衡的最终目的,是提升topic的并发消费能力，步骤如下：
1、 假如topic1,具有如下partitions: P0,P1,P2,P3
2、 加入group中,有如下consumer: C1,C2
3、 首先根据partition索引号对partitions排序: P0,P1,P2,P3
4、 根据consumer.id排序: C1,C2
5、 计算倍数: M = [P0,P1,P2,P3].size / [C1,C2].size,本例值M=2(向上取整)
6、 然后依次分配partitions: C1 = [P0,P3],C2=[P1,P2],即Ci = [P(i * M),P((i + 1) * M -1)] 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 030.png)

```
4.数据的分发策略
kafka默认会调用自己的分区器(DefaultPartition)进行分区,也可以自定义分区器,
自定义分区需要实现partitioner(特质),实现partition方法
key.hashcode() % numPartition
5.kafka是怎保证数据不丢失
镜像服务器,成本比较高,kafka接收数据数会创建topic指定副本的数量
副本的数据是由kafka自己进行同步的,多副本就保证了安全性
6.kafka可以保证topic里面的数据是全局有序吗?
kafka可以做到分区内有序,分区之间是无序
若需要做到全局有序,创建topic的时候总指定分区数
```

## Consumer与topic关系 

本质上kafka只支持Topic；
每个group中可以有多个consumer，每个consumer属于一个consumer group；
通常情况下，一个group中会包含多个consumer，这样不仅可以提高topic中消息的并发消费能力，而且还能提高"故障容错"性，如果group中的某个consumer失效那么其消费的partitions将会有其他consumer自动接管。
对于Topic中的一条特定的消息，只会被订阅此Topic的每个group中的其中一个consumer消费，此消息不会发送给一个group的多个consumer；那么一个group中所有的consumer将会交错的消费整个Topic，每个group中consumer消息消费互相独立，我们可以认为一个group是一个"订阅"者。
在kafka中,一个partition中的消息只会被group中的一个consumer消费(同一时刻)；
一个Topic中的每个partition，只会被一个"订阅者"中的一个consumer消费，不过一个consumer可以同时消费多个partition中的消息。
kafka的设计原理决定,对于一个topic，同一个group中不能有多于partition个数的consumer同时消费，否则将意味着某些consumer将无法得到消息。
**kafka只能保证一个partition中的消息被某个consumer消费时是顺序的；事实上，从Topic角度来说,当有多个partitions时,消息仍不是全局有序的。**

## Kafka消息的分发 

**Producer客户端负责消息的分发**
kafka集群中的任何一个broker都可以向producer提供metadata信息,这些metadata中包含"集群中存活的servers
列表"/"partitions leader列表"等信息；
当producer获取到metadata信息之后, producer将会和Topic下所有partition leader保持socket连接；
消息由producer直接通过socket发送到broker，中间不会经过任何"路由层"，事实上，消息被路由到哪个partition
上由producer客户端决定；
比如可以采用"random""key-hash""轮询"等,如果一个topic中有多个partitions,那么在producer端实现"消息均衡
分发"是必要的。
l在producer端的配置文件中,开发者可以指定partition路由的方式。
Producer消息发送的应答机制
设置发送数据是否需要服务端的反馈,有三个值0,1,-1
0: producer不会等待broker发送ack
1: 当leader接收到消息之后发送ack
-1: 当所有的follower都同步消息成功后发送ack
request.required.acks=0  

## 搭建集群 

需要搭建zookeeper和关闭防火墙
1.kafka上传安装包到hadoop01
2.解压到相对应的路径 tar -zxvf kafka_2.11-0.9.0.1.tgz -C /opt/software/
3.进入kafka安装路径的config目录下 cd /opt/software/kafka_2.11-0.9.0.1/config/
ps:因为配置过多不建议手写,所以提供配置文件代替,需要将原有的文件进行备份
4.将producer , server ,consumer进行备份(是原有系统文件没有被修改)
mv ./producer.properties ./producer.properties.bak mv ./server.properties ./server.properties.bak mv
./consumer.properties ./consumer.properties.bak
上传kafka配置文件
5.将目录进行分发
scp -r ./kafka_2.11-0.9.0.1/ root@hadoop03:$PWD
scp -r ./kafka_2.11-0.9.0.1/ root@hadoop04:$PWD
6.进入到分发过后的节点,进入kakfa安装路径中 config路径修改 server文件
需要修改broker.id 和 host.name
7.启动kafka集群(必须先启动zk)
进入kafka bin目录下进行启动
因为kafka前台启动不方便,启动到后台
nohup kafka-server-start.sh /opt/software/kafka_2.11-0.9.0.1/config/server.properties & 

## kafka的常用命令

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 020.png)

## kafka的API的形式模仿生产者和消费者

```scala
package day7

import java.util.Properties

import kafka.producer.{KeyedMessage, Producer, ProducerConfig}

/**
  *通过KafkaAPI实现生产者
  * 能够生产数据并发数据实时的发送kafka的某个topic中
  */
object KafkaProducerDemo {
  def main(args: Array[String]): Unit = {
    // 定义一个 topic
    val topic = "test"
    //创建一个配置类
    val prop = new Properties()
    //ps:这些配置都是config目录下produce.properties文件中的
    //key去这里查询  value 实际传入的值
    //指定序列化器
    prop.put("serializer.class","kafka.serializer.StringEncoder")
    //指定kafka集群列表
    prop.put("metadata.broker.list","hadoop1:9092,hadoop3:9092,hadoop4:9092")
   //设置发送数据后的响应方式
    prop.put("request.required.acks","1")
    //调用分区器
    prop.put("partitioner.class","kafka.producer.DefaultPartitioner")
    //自定义分区器
    // prop.put("partitioner.class","自定义分区器类的全限定名")
    //把配置信息封装到ProducerConfig中
     val config =  new ProducerConfig(prop)
    //ProducerConfig对象来创建Producer实例
    //创建producer对象的时候 指定泛型就是 topic的类型(代码定义)  最终要发送消息的数据类型(代码中定义)
    val p: Producer[String, String] = new Producer(config)
     //模拟数据产生
    for(i <- 1 to 10000){

      val msg = i+" Producer send data"
      //发送数据
      p.send(new KeyedMessage[String,String](topic,msg))
      //延迟 不然太过迅猛
      Thread.sleep(500)


    }
  }
}

```

```scala
package day7

import java.util.Properties
import java.util.concurrent.{Executor, Executors}

import kafka.consumer.{Consumer, ConsumerConfig, ConsumerConnector, KafkaStream}
import kafka.message.MessageAndMetadata

import scala.collection.mutable
//创建一个类这个类使用线程方式来处理数据
class KafkaConsumerDemo(val consumer:String, val stream:KafkaStream[Array[Byte], Array[Byte]]) extends Runnable{
  override def run(): Unit = {
    //转换成迭代器对象
      val it = stream.iterator()
    while(it.hasNext()){
      val data: MessageAndMetadata[Array[Byte], Array[Byte]] = it.next()
       val topic = data.topic
       val offset = data.offset
        val patition = data.partition
        val msg = new String(data.message())
      println("Consumer:"+consumer + topic + offset + patition +"msg:" + msg)
    }
  }
}
//模拟消费者
object KafkaConsumerDemo {
  def main(args: Array[String]): Unit = {
     // 读取topic的名称
     val topic = "test"

    //定义一个map,用来存储多个topic 第二个参数是使用多个线程来获取topic
    val topics =  new mutable.HashMap[String,Int]()
    topics.put(topic,2)

    //创建配置信息
    val prop = new Properties()
    //指定consumer组
    prop.put("group.id","group01")
    //指定zookeeper列表,获取数据的offset
    prop.put("zookeeper.connect","hadoop2:2181,hadoop3:2181,hadoop4:2181")
    /**
      * 如果zookeeper没有offset值或offset值超出范围。那么就给个初始的offset。有smallest、largest、anything可选，分别表示给当前最小的offset、当前最大的offset、抛异常。默认largest
      * auto.offset.reset=smallest
      */
    prop.put("auto.offset.reset","smallest")

    //创建ConsumerConfig对象
    val config: ConsumerConfig = new ConsumerConfig(prop)
    //创建Consumer实例
     val consumer: ConsumerConnector = Consumer.create(config)
    //获取数据 map的key是topic名称   value 获取来的数据
      val streams: collection.Map[String, List[KafkaStream[Array[Byte], Array[Byte]]]] = consumer.createMessageStreams(topics)

    //获取指定的topic的数据
    val stream: Option[List[KafkaStream[Array[Byte], Array[Byte]]]] = streams.get(topic)

    //创创建一个线程池
   val pool =  Executors.newFixedThreadPool(3)

    for(i <- 0 until stream.size){

      pool.execute(new KafkaConsumerDemo(i+"",stream.get(i)))
    }
   //kafka必须有 test 的 topic

  }
}
```

# day8

## 什么是DStream 

是一个离开散流,是SparkStreaming的基础抽象,代表持续性的数据流和经过各种spark原语((函数)方法)操作后得到的一个结果数据流,**DStream是一系列连续的RDD**所表示,每个RDD含有一段时间间隔内的数据,执行操作过程中把数据按照时间分割分成一个个批次进行处理
特性:
1.一个放了多个RDD的列表,而且DStream之间有依赖关系
2.每隔一段时间DStream就会生成一个RDD
3,每隔一段时间生成的RDD都有一个函数作用在这个RDD上 

## DStream的相关操作 

DStream的操作与RDD类似,分为Transformations(转换)和 OutputOperations(输出)和Window(窗口)
有一些较特殊的函数 updateStateByKey ,transform ,Window
1.UpdateStateByKey用于记录历史信息,对历史数据的一个获取和操作
2.Transform可以DStream中RDD执行本身的函数(DStream中的RDD获取出来进行操作 / 使用RDD本身的函数来操作RDD) 

## DStream转换操作 

| 操作                             | 含义                                                         |
| -------------------------------- | ------------------------------------------------------------ |
| map(func)                        | 通过传递源DStream的每个元素通过函数func返回一个新的DStream   |
| flatMap(func)                    | 与map类似，但每个输入项可以映射到0个或更多的输出项。         |
| filter(func)                     | 通过仅选择func返回true 的源DStream的记录来返回新的DStream    |
| repartition(numPartitions)       | 通过创建更多或更少的分区来更改此DStream中的并行级别。        |
| union(otherStream)               | 返回一个新的DStream，它包含源DStream和otherDStream中元素的并 集。 |
| count()                          | 通过计算源DStream的每个RDD中的元素数量来返回单元素RDD的新 DStream |
| reduce(func)                     | 通过使用函数func（它需要两个参数并返回一个），通过聚合源DStream的 每个RDD中的元素来返回单元素RDD的新DStream 。该函数应该是关联 的，以便可以并行计算。 |
| countByValue()                   | 当调用类型为K的元素的DStream时，返回一个新的DStream（K，Long） 对，其中每个键的值是源DStream的每个RDD中的频率。 |
| reduceByKey(func, [numTasks])    | 当（K，V）对的DStream被调用时，返回（K，V）对的新DStream，其中 使用给定的reduce函数聚合每个键的值。注意：默认情况下，使用Spark的 默认并行任务数（2为本地模式，群集模式中的数字由config属性决定 spark.default.parallelism）进行分组。您可以传递可选numTasks参数来 设置不同数量的任务。 |
| join(otherStream, [numTasks])    | 当（K，V）和（K，W）对的两个DStream被调用时，返回一个新的（K， （V，W））对的DStream与每个键的所有元素对。 |
| cogroup(otherStream, [numTasks]) | 当调用（K，V）和（K，W）对的DStream时，返回一个新的 DStream（K，Seq [V]，Seq [W]）元组。 |
| transform(func)                  | 通过对源DStream的每个RDD应用RDD到RDD函数来返回一个新的 DStream。这可以用于对DStream进行任意RDD操作。 |
| updateStateByKey(func)           | 返回一个新的“状态”DStream，其中通过对键的先前状态应用给定的功能和 键的新值来更新每个键的状态。这可以用于维护每个密钥的任意状态数据。 |

## DStream的输出操作

输出操作允许将DStream的数据推送到外部系统，如数据库或文件系统。由于输出操作实际上允许外部系统使用变换后的数据，所以它们触发所有DStream变换的实际执行（类似于RDD的动作）。目前，定义了以下输出操作： 

| 操作                                    | 含义                                                         |
| --------------------------------------- | ------------------------------------------------------------ |
| print（）                               | 在运行流应用程序的驱动程序节点上的DStream中打印每批数据的前十 个元素。 |
| saveAsTextFiles（prefix，[ suffix ]）   | 将此DStream的内容另存为文本文件。基于产生在每批间隔的文件名的 前缀和后缀：“前缀TIME_IN_MS [.suffix]”。 |
| saveAsObjectFiles（prefix，[ suffix ]） | 将DStream的内容保存为SequenceFiles序列化的Java对象。基于产生在 每批间隔的文件名的前缀和后缀：“前缀TIME_IN_MS [.suffix]”。 |
| saveAsHadoopFiles（prefix， [ suffix]） | 将此DStream的内容另存为Hadoop文件。基于产生在每批间隔的文件 名的前缀和后缀：“前缀TIME_IN_MS [.suffix]”。 |
| foreachRDD（func）                      | 对从流中生成的每个RDD 应用函数func的最通用的输出运算符。此功能 应将每个RDD中的数据推送到外部系统，例如将RDD保存到文件，或将 其通过网络写入数据库。请注意，函数func在运行流应用程序的驱动程 序进程中执行，通常会在其中具有RDD动作，从而强制流式传输RDD的 计算。 |

## 用SparkStreaming完成一个实时的WordCout 

SparkStreaming获取数据都是流式数据,从固定的介子中读取固定文件,不是太适合
使用到一个工具Linux下提供的netcat,需要创建一个端口 SparkStreaming可以进行监听获取数据
需要在任意一台节点上安装工具
yum install -y nc 启动netcat的时候启动一个client和Server,clinet会向Server发送信息
启动监听端口
nc -lk 6666
查看端口
netstat -nltp
写SparkStreaming代码(基础版本) 

```scala
package Day08
import org.apache.spark.streaming.dstream.{DStream, ReceiverInputDStream}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}
object SparkStreamingWCDemo {
def main(args: Array[String]): Unit = {
//先创建Sparkconf对象通过SparkConf对象创建SparkContext在通过SparkContext创建StreamingContext
对象
val conf = new SparkConf().setAppName("SparkStreamingWCDemo").setMaster("local[2]")
val sc = new SparkContext(conf)ps:
由于是本地模式,所以在运行时候需要指定并行度,这个并行度需要是>=2,因为是本地所以需要使用一个线程给
receiver,所以这个并行必须是>=2
若将任务提交到集群中,启动指定集群的cores,cores>=2
问题:每次通过服务端输入的数据,代码都可以正确的统计出来,但是数据不能累加不能完成实际业务需求
// 两个参数 第一个是 SparkContext对象, 第二个参数 批次时间间隔
val ssc = new StreamingContext(sc,Seconds(5));
//获取实时数据 从netcat服务器端获取数据
//这个方法也是SparkStreaming实时获取数据的方法
//修改过hosts文件传入名字即可 没有就是对应节点的IP
val dStream: ReceiverInputDStream[String] = ssc.socketTextStream("hadoop01",6666)
//处理数据DStream和RDD使用上是没有什么区别,含义是不同,可以直接使用操作RDD的方式来操作DStream
val sumed: DStream[(String, Int)] = dStream.flatMap(_.split("
")).map((_,1)).reduceByKey(_+_)
//将结果打印到控制台
sumed.print()
//开启任务(将任务提交到集群)
ssc.start()
//等待任务,处理下一个批次(线程等待)
ssc.awaitTermination()
}
}
```

ps:
由于是本地模式,所以在运行时候需要指定并行度,这个并行度需要是>=2,因为是本地所以需要使用一个线程给receiver,所以这个并行必须是>=2
若将任务提交到集群中,启动指定集群的cores,cores>=2
**问题:每次通过服务端输入的数据,代码都可以正确的统计出来,但是数据不能累加不能完成实际业务需求** 

```scala
package Day08
import org.apache.spark.{HashPartitioner, SparkConf}
import org.apache.spark.streaming.dstream.{DStream, ReceiverInputDStream}
import org.apache.spark.streaming.{Milliseconds, StreamingContext}
object SparkStreamingUpdateWCDemo {
//问题:每次通过服务端输入的数据,代码都可以正确的统计出来,但是数据不能累加不能完成实际业务需求
//psupdateStetaByKey只能拉取历史批次结果的数据到当前批次进行聚合操作
//但是并没有存储历史批次结果的功能,所以需要实现按批次累加就需要定义checkpoint
//sparkStreaming已经提供一个算子来处理这个累加问题 updateStateByKey
def main(args: Array[String]): Unit = {
val conf = new
SparkConf().setAppName("SparkStreamingUpdateWCDemo").setMaster("local[2]")
//第二种创建方式
//第一参数是conf对象 第二参数是时间间隔(批次处理)
// Seconds 秒 Millseconds 毫秒 之间差值是1000
val ssc = new StreamingContext(conf,Milliseconds(5000))
//定义检查点,用于存储历史批次数据
//可以存在本地,也可以存在HDFS,实际工作中是存储到HDFS上Transform操作
在是用SparkStreaming的时候,产生的数据集是一个DStream 一段批次内的RDD合集
DStream API无法满足需求: 将一段批次内的RDD 进行join若通过DstreamAPI来操作的话是无法完成
过滤案例:网站黑名单,某些IP禁止访问,对网站的广告点击量统计,某个IP在同一个时间段疯狂刷单,敏感字符串
ssc.checkpoint("out4")
// 获取监听数据
val dStream: ReceiverInputDStream[String] = ssc.socketTextStream("hadoop01",6666)
//将数据生成元组
val tuples: DStream[(String, Int)] = dStream.flatMap(_.split(" ")).map((_,1))
//第一个参数以DStream中数据key进行reduce,然后对各个批次的数据进行累加(自定义函数或自定义方法)
//第二个参数 需要分区数(默认分区 或 自定分区)
//第三个参数 表示是否在接下来SparkStreaming执行过程中产生的RDD使用相同分区算法
val sumed: DStream[(String, Int)] = tuples.updateStateByKey(func,new
HashPartitioner(ssc.sparkContext.defaultParallelism),true)
//打印
sumed.print()
//启动
ssc.start()
//等待
ssc.awaitTermination()
} /
/方法定义(Iterator[(K, Seq[V], Option[S])]) => Iterator[(K, S)]
//该函数中传入的参数是一个迭代器对象,数据类型都是泛型 启动K和S会影响后面返回值的数据类型
//第一个k一般是数据中的单词(key的数据类型)
//第二个seq[V]V是当前value中的数据 (当前批次相同key 对应的value(数字1)---seq(1,1,1,1))
//第三个Option[S] S是代表上一个批次相同key累加的结果 S的数据类型和V是一样的
//因为可能处理的时候是没有值的,所以这里使用Option
val func = (it:Iterator[(String,Seq[Int],Option[Int])])=>{
it.map(tup =>{
//Seq--> sum reduce ,若这里都是1调用size也是可以的
//seq(1,1,1,1) tup.sum 1+1+1+1 =4 tup.size = 4
(tup._1,tup._2.sum+tup._3.getOrElse(0))
})
}
}
```

## Transform操作 

在是用SparkStreaming的时候,产生的数据集是一个DStream 一段批次内的RDD合集
DStream API无法满足需求: 将一段批次内的RDD 进行join若通过DstreamAPI来操作的话是无法完成
过滤案例:网站黑名单,某些IP禁止访问,对网站的广告点击量统计,某个IP在同一个时间段疯狂刷单,敏感字符串 

```scala
package day8

import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.dstream.{DStream, ReceiverInputDStream}
import org.apache.spark.streaming.{Seconds, StreamingContext}
object TransformDemo {
  def main(args: Array[String]): Unit = {
      //创建模板对象
      val conf  = new SparkConf().setAppName("TransformDemo").setMaster("local[2]")
      val scc = new  StreamingContext(conf,Seconds(5))

    //获取数据资源
   val dStream: ReceiverInputDStream[String] = scc.socketTextStream("hadoop01",6666)
    //对数据进行一统计 生成元组
      val  wcDS = dStream.flatMap(_.split(" ")).map((_,1))
    //模拟一个黑名单
     val filters: RDD[(String, Boolean)] = scc.sparkContext.parallelize(List("ni","wo","ta")).map((_,true))
    //数据过滤使用transform算子对DStream中的RDD进行操作
    val words: DStream[(String, Int)] = wcDS.transform(rdd => {
      //合并
      val leftRDD: RDD[(String, (Int, Option[Boolean]))] = rdd.leftOuterJoin(filters)
      //过滤黑名单
      val word: RDD[(String, (Int, Option[Boolean]))] = leftRDD.filter(tuple => {
        val x = tuple._2
        if (x._2.isEmpty) {
          true
        } else {
          false
        }
      })
     // 过滤之后的结果
      word.map(tup => (tup._1, 1))
    })
    val sumed = words.reduceByKey(_+_)
    sumed.print()
    scc.start()
    scc.awaitTermination()
  }

}

```

## SparkStreaming整合Kafka 

执行操作:
1.启动zk
2,启动kafka
nohup kafka-server-start.sh /opt/software/kafka_2.11-0.9.0.1/config/server.properties &
创建topic
kafka-topics.sh --create --zookeeper hadoop02:2181 --replication-factor 1 --partitions 1 --topic wc
3.完成代码编写 

```scala
package day8

import org.apache.spark.streaming.dstream.{DStream, ReceiverInputDStream}
import org.apache.spark.streaming.kafka.KafkaUtils
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
object SparkStreamingAndKafka {
     //SaprkStreaming和Kafa整合
     def main(args: Array[String]): Unit = {
          val conf = new SparkConf().setAppName("SparkStreamingAndKafka").setMaster("local[2]")
          val sc = new SparkContext(conf)
          val ssc = new StreamingContext(sc,Seconds(5))

       //设置检查点:
         ssc.checkpoint("out5")
       //构建kafka
       //1.创建zk集群
       val zk = "hadoop02:2181,hadoop03:2181,hadoop04:2181"
       //2.消费组的组名
       val group = "group01"
       //因为可以有多个topic存在,所以需要创建一个Map存储所有的topic
       //key:topic名称  value 访问topic的线程数
       val topics = Map[String,Int](("wc",2))


       //调用 KafkaUtils工具来获取kafka中的数据
       //参数1: StreamingContext对象 参数2:zk集群  参数3:组名 参数4:存储所有topic的Map
       val dStream: ReceiverInputDStream[(String, String)] = KafkaUtils.createStream(ssc,zk,group,topics)

       //需要对数据进行操作
       //dStream中key是offset  value就是数据
       val lines: DStream[String] = dStream.map(_._2)
       //处理数据生成元组
       val tuples: DStream[(String, Int)] = lines.flatMap(_.split(" ")).map((_,1))
       //求和
       val sumed = tuples.updateStateByKey(func,new HashPartitioner(ssc.sparkContext.defaultParallelism),true)
       //打印
       sumed.print()
       //开始
       ssc.start()
       //等待
       ssc.awaitTermination()

     }
    val func =(it:Iterator[(String,Seq[Int],Option[Int])])=>{
       it.map{
        case(x,y,z) =>{
          (x,y.sum+z.getOrElse(0))
        }
      }
    }
}
```

模拟数据生产过程
在kafka集群中 启动一个节点作为 生产者
kafka-console-producer.sh --broker-list hadoop01:9092 --topic wc
创建数据: 

```
www.baidu.com
www.qfedu.com
www.csdn.com
www.hadoop.com
www.spark.com
www.hive.com
www.xxoo.com
```

## WindowOperations窗口操作 

SparkStreaming提供了滑动窗口操作支持,从而让我们可以对一个滑动窗口内的数据执行计算操作,每次出现在窗口内的RDD的数据会被聚合起来执行计算操作,然后生成新的RDD,新的RDD回座位windowDStream的一个RDD存在
ps:windowDStream一段RDD的集合 --> 窗口函数所产生 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 031.png)

上图是官网所示:
1.window size(窗口大小) 是 3 slied interval 移动的时间间隔 2
当前图片每隔2秒钟数据执行一次窗口滑动,这2秒内的3个RDD会被聚合处理,然后在过2秒钟会对近2秒测数据滑动窗口计算,每个滑动窗口的计算都必须指定两个参数, 窗口滑动的间隔(长度), 执行的批次间隔(时间)
其实窗口函数就是展示结果使用,可以展示固定时间内的数据
窗口操作的应用:
处理数据的时间间隔是5秒,展示数据也就是每5秒钟展示一次
但是现在希望可以1小时展示一次,此时可以调节间隔时间来进行数据处理,这样做可行吗?理论上是可以行,但是会出现一个问题会造成一段时间内数据大量的堆积,计算成本就要增加可以使用窗口函数获取这段时间内的数据计算并展示出来,每隔一段时间向后滑动获取新的数据 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 032.png)

| 操作                                                         | 含义                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| window（windowLength，slideInterval）                        | 返回基于源DStream的窗口批次计算的新DStream。                 |
| countByWindow（windowLength， slideInterval）                | 返回流中元素的滑动窗口数。                                   |
| reduceByWindow（func，windowLength， slideInterval）         | 返回一个新的单元素流，通过使用func在滑动间隔中通 过在流中聚合元素创建。 |
| reduceByKeyAndWindow（func， windowLength，slideInterval，[ numTasks ]） | 当对（K，V）对的DStream进行调用时，返回（K，V） 对的新DStream，其中每个键的值 在滑动窗口中使用给 定的减少函数func进行聚合。 |
| countByValueAndWindow（windowLength， slideInterval，[numTasks ]） | 当调用（K，V）对的DStream时，返回（K，Long）对 的新DStream，其中每个键的值是其滑动窗口内的频率。 |

```scala
package Day08
import org.apache.spark.SparkConf
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.streaming.{Milliseconds, StreamingContext}
object WindowDemo {
def main(args: Array[String]): Unit = {
val conf = new SparkConf().setAppName("WindowDemo").setMaster("local[2]")
//执行批次的时间是5秒
val ssc = new StreamingContext(conf,Milliseconds(5000))
val dStream = ssc.socketTextStream("hadoop01",6666)
val tuples = dStream.flatMap(_.split(" ")).map((_,1))
// 这个函数是不能使用_的 窗口大小(10秒)即展示范围 2个批次的数据 滑动时间的间隔 10秒
//展示的其实就是2个批次的数据
val sumed: DStream[(String, Int)] = tuples.reduceByKeyAndWindow((x:Int, y:Int)=>
(x+y),Milliseconds(10000),Milliseconds(10000))
sumed.print()
ssc.start()
ssc.awaitTermination()
}
}
```

# day9

## spark源代码解析







# day10

## spark的Shuffle过程 

shuffle操作就是在spark操作中调用了一些特殊的算子才触发的一种操作
shuffle操作会导致大量的数据在不同的节点之间传输因此,shuffle过程是Spark中最复杂,最耗时,最消耗性能的一种操作
reduceByKey算子会将上一个RDD中每个key 对应的所有value都聚合在一个value,生成一个新的RDD
先的RDD与元素类型即使的格式,每个Key对应一个聚合起来的value上一个RDD中会有多个partition会影响下一个RDD的partition,并且会根据相同key将数据拉去到同一个partition触发的过程就会就会发生shuffle,这个过程势必会造成大量的网络IO
SparkShuffle分为两大类HashShuffle 和 SortShuffle
SparkShuffle 分两个过程 shuffle write 和 shuffle read 而且是在不同 stage中进行的
HashShuffle 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 049.png)

****在Spark1.2以前默认使用的是Shuffle计算引擎是HashShuffle,因为HashShuffle会产生大量的磁盘小文件从而导致新能下降,在spark1.2之后版本中,默认的HashShuffle被修改后才能SortShuffle
**SortShuffle相当于HashShuffle来说,有一定的改进,主要是在于,每个Task在进行shuffle的时候,虽然会产生很多临时的磁盘文件,但是最后会将临时的磁盘文件进行合并(merge)成一个磁盘文件,一位每个Task只有有一个文件,在一下一个stage拉取数据的时候明,只要根据索引就可以获取磁盘中的文件**
**SortShuffle运行机制主要分为两种,一种是普通机制,另外一种是bypass运行机制**
当shuffle read读取task的数量小于等于 200(默认值),就会启动bypass机制** 

![](D:\文件下载路径\电脑默认图片\Saved Pictures\图像 050.png)

shuffle过程中分区排序问题
默认情况下,shuffle操作是不会对分区中的数据进行排序操作
如果想要对每个分区中的数据进行排序,可以使用三种方法
1. 使用mapPartitions算子把每个Partition中的数据进行一个排序
2. 使用repartitionAndSortWithInPartitions(该算子是对RDD进行重新分区的算子),在重分区的过程中会对分区中的数据进行重新排序
3. 使用sortByKey 对所有分区数据进行全局排序
  ps:若需要分区排序mapPartitions代价最小,因为不需要额外的shuffle操作
  repartitionAndSortWithInPartitions和sortByKey都需要额外的shuffle,新能并不高
  会导致shuffle的算子
  byKey的: reduceByKey,grouoByKey,sortByKey.aggregateByKey,combineByKey
  repartition的: repartition,repartitionAndSortWithInPartitions,coalesce
  Join类算子:若先使用groupBykey后在使用 join就不会发生shuffle
  ps:尽量避免shuffle因为shuffle会造成大量的IO 

## shuffle的调优 

| 属性名称                               | 默认 值 | 属性说明                                                     |
| -------------------------------------- | ------- | ------------------------------------------------------------ |
| spark.reducer.maxSizeInFlight          | 48m     | reduce task的buffer缓冲，代表了每个reduce task每次能够拉取的map side数据最大大小， 如果内存充足，可以考虑加大，从而减少网络传 输次数，提升性能 |
| spark.shuffle.blockTransferService     | netty   | shuffle过程中，传输数据的方式，两种选项， netty或nio，spark 1.2开始，默认就是netty， 比较简单而且性能较高，spark 1.5开始nio就是 过期的了，而且spark 1.6中会去除掉 |
| spark.shuffle.compress                 | true    | 是否对map side输出的文件进行压缩，默认是 启用压缩的，压缩器是由 spark.io.compression.codec属性指定的，默认 是snappy压缩器，该压缩器强调的是压缩速 度，而不是压缩率 |
| spark.shuffle.consolidateFiles         | false   | 默认为false，如果设置为true，那么就会合并 map side输出文件，对于reduce task数量特别 的情况下，可以极大减少磁盘IO开销，提升性能 |
| spark.shuffle.file.buffer              | 32k     | map side task的内存buffer大小，写数据到磁 盘文件之前，会先保存在缓冲中，如果内存充 足，可以适当加大，从而减少map side磁盘IO 次数，提升性能 |
| spark.shuffle.io.maxRetries            | 3       | 网络传输数据过程中，如果出现了网络IO异常， 重试拉取数据的次数，默认是3次，对于耗时的 shuffle操作，建议加大次数，以避免full gc或者 网络不通常导致的数据拉取失败，进而导致task lost，增加shuffle操作的稳定性 |
| spark.shuffle.io.retryWait             | 5s      | 每次重试拉取数据的等待间隔，默认是5s，建议 加大时长，理由同上，保证shuffle操作的稳定 性 |
| spark.shuffle.io.numConnectionsPerPeer | 1       | 机器之间的可以重用的网络连接，主要用于在大 型集群中减小网络连接的建立开销，如果一个集 群的机器并不多，可以考虑增加这个值 |
| spark.shuffle.io.preferDirectBufs      | true    | 启用堆外内存，可以避免shuffle过程的频繁 gc，如果堆外内存非常紧张，则可以考虑关闭这 个选项 |

| 属性名称                                | 默认 值 | 属性说明                                                     |
| --------------------------------------- | ------- | ------------------------------------------------------------ |
| spark.shuffle.manager                   | sort    | ShuffleManager，Spark 1.5以后，有三种可选 的，hash、sort和tungsten-sort，sort-based ShuffleManager会更高效实用内存，并且避免 产生大量的map side磁盘文件，从Spark 1.2开 始就是默认的选项，tungsten-sort与sort类似， 但是内存性能更高 |
| spark.shuffle.memoryFraction            | 0.2     | 如果spark.shuffle.spill属性为true，那么该选项 生效，代表了executor内存中，用于进行 shuffle reduce side聚合的内存比例，默认是 20%，如果内存充足，建议调高这个比例，给 reduce聚合更多内存，避免内存不足频繁读写 磁盘 |
| spark.shuffle.service.enabled           | false   | 启用外部shuffle服务，这个服务会安全地保存 shuffle过程中，executor写的磁盘文件，因此 executor即使挂掉也不要紧，必须配合 spark.dynamicAllocation.enabled属性设置为 true，才能生效，而且外部shuffle服务必须进行 安装和启动，才能启用这个属性 |
| spark.shuffle.service.port              | 7337    | 外部shuffle服务的端口号，具体解释同上                        |
| spark.shuffle.sort.bypassMergeThreshold | 200     | 对于sort-based ShuffleManager，如果没有进 行map side聚合，而且reduce task数量少于这 个值，那么就不会进行排序，如果你使用sort ShuffleManager，而且不需要排序，那么可以 考虑将这个值加大，直到比你指定的所有task数 量都打，以避免进行额外的sort，从而提升性能 |
| spark.shuffle.spill                     | true    | 当reduce side的聚合内存使用量超过了 spark.shuffle.memoryFraction指定的比例时， 就进行磁盘的溢写操作 |
| spark.shuffle.spill.compress            | true    | 同上，进行磁盘溢写时，是否进行文件压缩，使 用spark.io.compression.codec属性指定的压缩 器，默认是snappy，速度优先 |

## Shuffle过程对新能消耗的描述 

shuffle操作是Spark中唯一最消耗性能的过程
因此在需要的地方需要进行性能调优,上线之后的报错,出现数据倾斜
shuffle操作的前部分属于上一个stage的范围, 通常称为map task
shuffle操作的后部分属于下一个stage的范围,通常称为reduce task
其中map task负责数据的组织,也就是相同key对应的value写入桶一个下游task对应的分区文件中
其中reduce task 负责数据的聚合,也即是将上一个stage的task所在的节点上,将数据自己的数据拉去过来进行 聚合
map task将数据先保存在内存中,若内存不够会在溢写到磁盘文件
readuc task会读取各个基点上属于自己的数据并来过来进行聚合
由此可见:shuffle操作会消耗大量的内存,无论是网络传输是在之前(map)之后(reduce)都会进行大量的IO读写 

## Spark2.x 

看文档 

## 开发Spark2.x

## 老版本API 

```scala
/*
老版本API
*/
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
object OldSparkWordCount {
def main(args: Array[String]): Unit = {
val conf = new SparkConf().setAppName("OldSparkWordCount").setMaster("local[2]")
val sc = new SparkContext(conf)
val lines: RDD[String] = sc.textFile("dir/file.txt")
val word: RDD[String] = lines.flatMap(_.split(" "))
val tuples: RDD[(String, Int)] = word.map((_,1))
val sumed: RDD[(String, Int)] = tuples.reduceByKey(_+_)
val sorted: RDD[(String, Int)] = sumed.sortBy(_._2,false)
print(sorted.collect.toList)
sorted.saveAsTextFile("out")
sc.stop()
}
}
```

## DataFrame入门

```scala
/
**
DataFrame入门
*/
import org.apache.spark.sql.{DataFrame, SparkSession}
object NewDataFrame {
def main(args: Array[String]): Unit = {
//先构建sparkSession对象
//builder 构建SparkSession getOrCreate创建sparkSession
//通过builder构建配置数据,getOrCreate 创建SpareSession对象
val spark =
SparkSession.builder().appName("NewDataFrame").master("local[2]").getOrCreate()
//SparkSql读取文件的默认类型是什么?parquet格式
//读取数据
val dataFarme: DataFrame = spark.read.json("dir/people.json")
dataFarme.show()
dataFarme.printSchema()//DSL语言风格
dataFarme.select("name","age").show()
dataFarme.filter("age>20").show()
//隐式导入
import spark.implicits._
//让age都增加1岁使用 $ 这个语法 这个是取值操作
dataFarme.select($"name",$"age"+1,$"facevalue").show()
//用SQL语法风格进行查询
dataFarme.createOrReplaceTempView("t_people") //生成临时表
val sqldf: DataFrame = spark.sql("select * from t_people")
sqldf.show()
}
}
```

## DataSet入门

```scala
/
**
DataSet入门
*/
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
object DataSetDemo {
def main(args: Array[String]): Unit = {
val spark =
SparkSession.builder().appName("DataSetDemo").master("local").getOrCreate()
import spark.implicits._
//DataSet构建有两种
// 一种就直接构建 其实可以当做是SQL版本的RDD使用
val personDS: Dataset[Person] = Seq(Person("xiaode",20,50)).toDS()
personDS.show()
// 使用到DataFrame构建
val df: DataFrame = spark.read.json("dir/people.json")
//把DataFrame构建成DataSet
val peopleDS: Dataset[Person] = df.as[Person]
peopleDS.show()
spark.stop()
}
} 
//Dataset或DataFrame也好 在创建样例类进行数据匹配或存储的时候不要在使用Int 而是使用Long
case class Person(name:String,age:Long,faceValue:Long)
```

## hive入门

```scala
/**
hive入门
*/
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}object NewHiveContext {
def main(args: Array[String]): Unit = {
val spark = SparkSession.builder()
.appName("NewHiveContext")
.master("local")
//设置sparksql的元数据仓库的目录
.config("spark.sql.warehouse.dir","D://spark-warehouse")
//启动hive支持
.enableHiveSupport()
.getOrCreate()
//导入隐式转换
import spark.implicits._
//创建一张hive表
//spark.sql("create table if not exists src(key int,value string)row format delimited
fields terminated by ' '")
//加载数据
//spark.sql("load data local inpath 'dir/kv1.txt' into table src")
//查询表中的内容
// spark.sql("select * from src").show()
//统计条数
//spark.sql("select count(*) from src").show()
val sqlHiveDF: DataFrame = spark.sql("select key,value from src where key<=10 order
by key")
//价格DataFrame转换为DataSet
//Dataset[Row] Row和Any差不多可以接受任何数据类型
val sqlHiveDS: Dataset[String] = sqlHiveDF.map {
case Row(key: Int, value: String) => s"key:$key,value:$value"
} 
sqlHiveDS.show()
spark.stop()
}
}
```

```scala
/
**
练习
*/
import org.apache.spark.sql.{DataFrame, SparkSession}
object Exercise {
/**
* 统计部分的平均薪水和平均年龄
* 需求:
* 1.只统计年龄在20以上的员工
* 2.根据部门名称和员工性别作为统计依据
* 3.统计出每个部门的平均薪水和平均年龄
* @param args
*/
def main(args: Array[String]): Unit = {
val spark =
SparkSession.builder().appName("Exercise").master("local").getOrCreate()
//导入隐式转换import spark.implicits._
//使用sparksql中function函数
import org.apache.spark.sql.functions._
//加载数据
//员工
val employee: DataFrame = spark.read.json("dir/employee.json")
//部门
val department: DataFrame = spark.read.json("dir/department.json")
//开始处理数据
employee
//对员工信息进行过滤,只统计大于20岁员工
.filter("age>20" )
//做join操作 把部门信息和员工信息join再一次 参数是要进行join的数据和 join的依据
//表示两个表的连接条件需要使用 ===
.join(department,$"depId" === $"id")
//根据部门名称和员工性别进行分组操作
.groupBy(department("name"),employee("gender"))
//最后的聚合
.agg(avg(employee("salary")),avg(employee("age")))
.show()
spark.stop()
}
} 
/
* 文件
输出
*/
object OutputTableDemo {
def main(args: Array[String]): Unit = {
val spark =
SparkSession.builder().appName("OutputTableDemo").master("local").getOrCreate()
//获取数据
val df: DataFrame = spark.read.json("dir/employee.json")
//创建一个临时表
df.createOrReplaceTempView("t_employee")
//查询
val edf: DataFrame = spark.sql("select * from t_employee where age > 25")
//存储
edf.write.mode("append").csv("out1")
}
}
```

## spark2.x算子

```scala
/
* 
spark2.x算子
*/
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}object Spark2xOperator {
//Spark2.x的算子
def main(args: Array[String]): Unit = {
val spark =
SparkSession.builder().appName("Spark2xOperator").master("local").getOrCreate()
//导入隐式转换
import spark.implicits._
//1.重新分区 repartition coalesce
val employee: DataFrame = spark.read.json("dir/employee.json")
//打印分区个数
println("初始化分区为:"+ employee.rdd.partitions.length)
//更改分区repartition可以从少量分区改变为多分区因为会发生shuffle
val Erepartition: Dataset[Row] = employee.repartition(3)
println("通过repartiton调整分区后的分区:"+ Erepartition.rdd.partitions.length)
//更改分区 不可以少量分区更改为多分区,因为不会发生shuffle
val Ecoalesce: Dataset[Row] = employee.coalesce(3)
println("通过coalesce调整分区后的分区:"+ Ecoalesce.rdd.partitions.length)
//去重(整条数据去重需要完全一样)
employee.distinct().show()
println("--------------------------------------------------静静分割线----------------
----------------------")
//去重(以数据中某个字段去重)
employee.dropDuplicates(Array("name")).show()
//过滤
val employee1 = spark.read.json("dir/employee.json")
val employee2 = spark.read.json("dir/employee2.json")
//将DataFrame转换为DataSet
val employeeDS1 = employee1.as[Employee]
val employeeDS2 = employee2.as[Employee]
//filter过滤
println("--------------------------------------------------静静分割线----------------
----------------------")
employeeDS1.filter(e => e.age > 30).show()
println("--------------------------------------------------静静分割线----------------
----------------------")
//根据启动一个DataSet中数据来过滤另外一个DataSet
employeeDS1.except(employeeDS2).show()
println("--------------------------------------------------静静分割线----------------
----------------------")
// 获取交集
(employeeDS1 intersect employeeDS2).show()
//合并
val department = spark.read.json("dir/department.json")
val departmentDS = department.as[Department]
//joinwith ,必须指定那个表和用那个字段 ps建议使用这个算子做合并 可以避免笛卡尔积
val joined: Dataset[(Employee, Department)] =
employeeDS1.joinWith(departmentDS,$"depId" === $"id")
println("--------------------------------------------------静静分割线----------------
----------------------")
joined.show()//排序 sort 指定那个字段来进行排序
println("--------------------------------------------------静静分割线----------------
----------------------")
employeeDS1.sort($"salary".desc).show()
//随机
//将一个DataSet分成过个DataSet
//第一个代表分配的权重 参数是一个Double
//第二个参数是一个种子,不填写
val arr: Array[Dataset[Employee]] = employeeDS1.randomSplit(Array(0.2,0.8))
println("--------------------------------------------------静静分割线----------------
----------------------")
arr.foreach(ds => ds.show())
//采样 随机按照一定的比抽取数据
//建议不同的采样器:false是伯努利分布(元素可以多次采样) true是泊松分布
//第二个参数 采样的比例
println("--------------------------------------------------静静分割线----------------
----------------------")
employeeDS1.sample(false,0.9).show()
//常用类的聚合
//avg平均数 sum 求和 max 最大值 min 最小 count技术 countDistinct 计数并去重 agg集合函数
//除agg外需要使用删除类SQL的函数都需要进行导包
import org.apache.spark.sql.functions._
println("--------------------------------------------------静静分割线----------------
----------------------")
employee.join(department,$"depId"===$"id").groupBy(department("name"))
.agg(avg(employee("salary")),sum(employee("salary")),max(employee("salary")),
min(employee("salary")),count(employee("name")),countDistinct(employee("name"))).show()
//转换集合
//collect_list(不去重) 和 collect_set(去重)
//常用语行转列
println("--------------------------------------------------静静分割线----------------
----------------------")
employee.groupBy(employee("depId"))
.agg(collect_list("name"),collect_set("name")).collect().foreach(println)
/*
日期函数 current_date(年月日) current_timestamp(年月日时分秒)
数学函数 round 保留几位小数
随机函数 rand
字符串函数 concat concat_ws(可以多拼接字符 ,可以以某个字符作为连接)
*/
println("--------------------------------------------------静静分割线----------------
----------------------")
employee.select(employee("name"),current_date(),current_timestamp(),rand()
,round(employee("salary"),2),concat(employee("gender"),employee("age"))
,concat_ws("|"),employee("gender"),employee("age")).show()
spark.stop()}
} 
case class Employee(name:String,age:Long,depId:Long,gender:String,salary:Double)
case class Department(id:Long,name:String)
```

## StructuredStreamingDemo

```scala
/*
StructuredStreamingDemo
*/
import org.apache.spark.sql.streaming.StreamingQuery
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
object StructuredStreamingDemo {
//当前Streaming是一个完全的流式处理
//省略了批次间隔并且只需要一个也能执行
def main(args: Array[String]): Unit = {
val spark =
SparkSession.builder().appName("StructuredStreamingDemo").master("local").getOrCreate()
//获取数据--> netcat 数据类型依旧是一个DataFrame
val lines =
spark.readStream.format("socket").option("host","hadoop01").option("port",9999).load()
//将数据进行转换DataFrame 转换 DataSet
val linesDS: Dataset[String] = lines.as[String]
//拆分数据
val words: Dataset[String] = linesDS.flatMap(_.split(" "))
//求和
val sumed: DataFrame = words.groupBy("value").count()
//将数据打印到控制台上
val query: StreamingQuery =
sumed.writeStream.outputMode("complete").format("console").start()
query.awaitTermination()
// nc -lk 9999
}
}
```

# 大数据spark日志系统项目

## note1

```
课程整套CDH相关的软件下载地址：http://archive.cloudera.com/cdh5/cdh/5/
cdh-5.7.0
生产或者测试环境选择对应CDH版本时，一定要采用尾号是一样的版本



http://hadoop.apache.org/
对于Apache的顶级项目来说，projectname.apache.org
Hadoop: hadoop.apache.org
Hive: hive.apache.org
Spark: spark.apache.org
HBase: hbase.apache.org


为什么很多公司选择Hadoop作为大数据平台的解决方案？
1）源码开源
2）社区活跃、参与者很多  Spark
3）涉及到分布式存储和计算的方方面面： 
   Flume进行数据采集
   Spark/MR/Hive等进行数据处理
   HDFS/HBase进行数据存储
4) 已得到企业界的验证



HDFS架构

1 Master(NameNode/NN)  带 N个Slaves(DataNode/DN)
HDFS/YARN/HBase

1个文件会被拆分成多个Block
blocksize：128M
130M ==> 2个Block： 128M 和 2M

NN：
1）负责客户端请求的响应
2）负责元数据（文件的名称、副本系数、Block存放的DN）的管理

DN：
1）存储用户的文件对应的数据块(Block)
2）要定期向NN发送心跳信息，汇报本身及其所有的block信息，健康状况

A typical deployment has a dedicated machine that runs only the NameNode software. 
Each of the other machines in the cluster runs one instance of the DataNode software.
The architecture does not preclude running multiple DataNodes on the same machine 
but in a real deployment that is rarely the case.

NameNode + N个DataNode
建议：NN和DN是部署在不同的节点上


replication factor：副本系数、副本因子

All blocks in a file except the last block are the same size




本课程软件存放目录
hadoop/hadoop
/home/hadoop
   software: 存放的是安装的软件包
   app : 存放的是所有软件的安装目录
   data: 存放的是课程中所有使用的测试数据目录
   source: 存放的是软件源码目录，spark


Hadoop环境搭建
1) 下载Hadoop
   http://archive.cloudera.com/cdh5/cdh/5/
   2.6.0-cdh5.7.0

   wget http://archive.cloudera.com/cdh5/cdh/5/hadoop-2.6.0-cdh5.7.0.tar.gz

2）安装jdk
   下载
   解压到app目录：tar -zxvf jdk-7u51-linux-x64.tar.gz -C ~/app/
   验证安装是否成功：~/app/jdk1.7.0_51/bin      ./java -version
   建议把bin目录配置到系统环境变量(~/.bash_profile)中
      export JAVA_HOME=/home/hadoop/app/jdk1.7.0_51
      export PATH=$JAVA_HOME/bin:$PATH


3）机器参数设置
   hostname: hadoop001
   
   修改机器名: /etc/sysconfig/network
      NETWORKING=yes
      HOSTNAME=hadoop001

   设置ip和hostname的映射关系: /etc/hosts
      192.168.199.200 hadoop001
      127.0.0.1 localhost

   ssh免密码登陆(本步骤可以省略，但是后面你重启hadoop进程时是需要手工输入密码才行)
      ssh-keygen -t rsa
      cp ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys

4）Hadoop配置文件修改: ~/app/hadoop-2.6.0-cdh5.7.0/etc/hadoop
   hadoop-env.sh
      export JAVA_HOME=/home/hadoop/app/jdk1.7.0_51

   core-site.xml
      <property>
           <name>fs.defaultFS</name>
           <value>hdfs://hadoop001:8020</value>
       </property>    

       <property>
           <name>hadoop.tmp.dir</name>
           <value>/home/hadoop/app/tmp</value>
       </property>    

    hdfs-site.xml
       <property>
           <name>dfs.replication</name>
           <value>1</value>
       </property>

5）格式化HDFS
   注意：这一步操作，只是在第一次时执行，每次如果都格式化的话，那么HDFS上的数据就会被清空
   bin/hdfs namenode -format

6）启动HDFS
   sbin/start-dfs.sh

   验证是否启动成功:
      jps
         DataNode
         SecondaryNameNode
         NameNode

      浏览器
         http://hadoop001:50070/


7）停止HDFS
   sbin/stop-dfs.sh





YARN架构
1 RM(ResourceManager) + N NM(NodeManager)

ResourceManager的职责： 一个集群active状态的RM只有一个，负责整个集群的资源管理和调度
1）处理客户端的请求(启动/杀死)
2）启动/监控ApplicationMaster(一个作业对应一个AM)
3）监控NM
4）系统的资源分配和调度


NodeManager：整个集群中有N个，负责单个节点的资源管理和使用以及task的运行情况
1）定期向RM汇报本节点的资源使用请求和各个Container的运行状态
2）接收并处理RM的container启停的各种命令
3）单个节点的资源管理和任务管理

ApplicationMaster：每个应用/作业对应一个，负责应用程序的管理
1）数据切分
2）为应用程序向RM申请资源(container)，并分配给内部任务
3）与NM通信以启停task， task是运行在container中的
4）task的监控和容错

Container：
对任务运行情况的描述：cpu、memory、环境变量

YARN执行流程
1）用户向YARN提交作业
2）RM为该作业分配第一个container(AM)
3）RM会与对应的NM通信，要求NM在这个container上启动应用程序的AM
4) AM首先向RM注册，然后AM将为各个任务申请资源，并监控运行情况
5）AM采用轮训的方式通过RPC协议向RM申请和领取资源
6）AM申请到资源以后，便和相应的NM通信，要求NM启动任务
7）NM启动我们作业对应的task



YARN环境搭建
mapred-site.xml
   <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>

yarn-site.xml
   <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>

启动yarn：sbin/start-yarn.sh

验证是否启动成功
   jps
      ResourceManager
      NodeManager

   web: http://hadoop001:8088

停止yarn： sbin/stop-yarn.sh

提交mr作业到yarn上运行： wc

/home/hadoop/app/hadoop-2.6.0-cdh5.7.0/share/hadoop/mapreduce/hadoop-mapreduce-examples-2.6.0-cdh5.7.0.jar

hadoop jar /home/hadoop/app/hadoop-2.6.0-cdh5.7.0/share/hadoop/mapreduce/hadoop-mapreduce-examples-2.6.0-cdh5.7.0.jar wordcount /input/wc/hello.txt /output/wc/

当我们再次执行该作业时，会报错：
FileAlreadyExistsException: 
Output directory hdfs://hadoop001:8020/output/wc already exists



Hive底层的执行引擎有：MapReduce、Tez、Spark
   Hive on MapReduce
   Hive on Tez
   Hive on Spark

压缩：GZIP、LZO、Snappy、BZIP2..
存储：TextFile、SequenceFile、RCFile、ORC、Parquet
UDF：自定义函数



Hive环境搭建
1）Hive下载：http://archive.cloudera.com/cdh5/cdh/5/
   wget http://archive.cloudera.com/cdh5/cdh/5/hive-1.1.0-cdh5.7.0.tar.gz

2）解压
   tar -zxvf hive-1.1.0-cdh5.7.0.tar.gz -C ~/app/

3）配置
   系统环境变量(~/.bahs_profile)
      export HIVE_HOME=/home/hadoop/app/hive-1.1.0-cdh5.7.0
      export PATH=$HIVE_HOME/bin:$PATH

   实现安装一个mysql， yum install xxx

   hive-site.xml
   <property>
      <name>javax.jdo.option.ConnectionURL</name>
       <value>jdbc:mysql://localhost:3306/sparksql?createDatabaseIfNotExist=true</value>
    </property>
    
   <property>
       <name>javax.jdo.option.ConnectionDriverName</name>
        <value>com.mysql.jdbc.Driver</value>
       </property>

   <property>
      <name>javax.jdo.option.ConnectionUserName</name>
       <value>root</value>
    </property>

   <property>
      <name>javax.jdo.option.ConnectionPassword</name>
       <value>root</value>
    </property>

4）拷贝mysql驱动到$HIVE_HOME/lib/

5）启动hive: $HIVE_HOME/bin/hive


创建表
CREATE  TABLE table_name 
  [(col_name data_type [COMMENT col_comment])]
  

create table hive_wordcount(context string);

加载数据到hive表
LOAD DATA LOCAL INPATH 'filepath' INTO TABLE tablename 

load data local inpath '/home/hadoop/data/hello.txt' into table hive_wordcount;


select word, count(1) from hive_wordcount lateral view explode(split(context,'\t')) wc as word group by word;

lateral view explode(): 是把每行记录按照指定分隔符进行拆解

hive ql提交执行以后会生成mr作业，并在yarn上运行


create table emp(
empno int,
ename string,
job string,
mgr int,
hiredate string,
sal double,
comm double,
deptno int
) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t';

create table dept(
deptno int,
dname string,
location string
) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t';

load data local inpath '/home/hadoop/data/emp.txt' into table emp;
load data local inpath '/home/hadoop/data/dept.txt' into table dept;

求每个部门的人数
select deptno, count(1) from emp group by deptno;
```



## note2

```
MapReduce的局限性：
1）代码繁琐；
2）只能够支持map和reduce方法；
3）执行效率低下；
4）不适合迭代多次、交互式、流式的处理；

框架多样化：
1）批处理（离线）：MapReduce、Hive、Pig
2）流式处理（实时）： Storm、JStorm
3）交互式计算：Impala

学习、运维成本无形中都提高了很多

===> Spark 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BDAS:Berkeley Data Analytics Stack

```

## note3

```
前置要求：
1）Building Spark using Maven requires Maven 3.3.9 or newer and Java 7+
2）export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m"

mvn编译命令：
./build/mvn -Pyarn -Phadoop-2.4 -Dhadoop.version=2.4.0 -DskipTests clean package
	前提：需要对maven有一定的了解(pom.xml)

<properties>
    <hadoop.version>2.2.0</hadoop.version>
    <protobuf.version>2.5.0</protobuf.version>
    <yarn.version>${hadoop.version}</yarn.version>
</properties>

<profile>
  <id>hadoop-2.6</id>
  <properties>
    <hadoop.version>2.6.4</hadoop.version>
    <jets3t.version>0.9.3</jets3t.version>
    <zookeeper.version>3.4.6</zookeeper.version>
    <curator.version>2.6.0</curator.version>
  </properties>
</profile>

./build/mvn -Pyarn -Phadoop-2.6 -Phive -Phive-thriftserver -Dhadoop.version=2.6.0-cdh5.7.0 -DskipTests clean package

#推荐使用
./dev/make-distribution.sh --name 2.6.0-cdh5.7.0 --tgz  -Pyarn -Phadoop-2.6 -Phive -Phive-thriftserver -Dhadoop.version=2.6.0-cdh5.7.0



编译完成后：
spark-$VERSION-bin-$NAME.tgz

spark-2.1.0-bin-2.6.0-cdh5.7.0.tgz




Spark Standalone模式的架构和Hadoop HDFS/YARN很类似的
1 master + n worker


spark-env.sh
SPARK_MASTER_HOST=hadoop001
SPARK_WORKER_CORES=2
SPARK_WORKER_MEMORY=2g
SPARK_WORKER_INSTANCES=1


hadoop1 : master
hadoop2 : worker
hadoop3 : worker
hadoop4 : worker
...
hadoop10 : worker

slaves:
hadoop2
hadoop3
hadoop4
....
hadoop10

==> start-all.sh   会在 hadoop1机器上启动master进程，在slaves文件配置的所有hostname的机器上启动worker进程


Spark WordCount统计
val file = spark.sparkContext.textFile("file:///home/hadoop/data/wc.txt")
val wordCounts = file.flatMap(line => line.split(",")).map((word => (word, 1))).reduceByKey(_ + _)
wordCounts.collect

```

## note4

```
文本文件进行统计分析：
id, name, age, city
1001,zhangsan,45,beijing
1002,lisi,35,shanghai
1003,wangwu,29,tianjin
.......

table定义：person
column定义：
	id：int
	name：string
	age： int
	city：string
hive：load data


sql: query....



Hive: 类似于sql的Hive QL语言， sql==>mapreduce
	特点：mapreduce
	改进：hive on tez、hive on spark、hive on mapreduce

Spark: hive on spark ==> shark(hive on spark)
	shark推出：欢迎， 基于spark、基于内存的列式存储、与hive能够兼容
	缺点：hive ql的解析、逻辑执行计划生成、执行计划的优化是依赖于hive的
		仅仅只是把物理执行计划从mr作业替换成spark作业


Shark终止以后，产生了2个分支：
1）hive on spark
	Hive社区，源码是在Hive中
2）Spark SQL
	Spark社区，源码是在Spark中
	支持多种数据源，多种优化技术，扩展性好很多



SQL on Hadoop
1）Hive 
	sql ==> mapreduce
	metastore ： 元数据 
	sql：database、table、view
	facebook

2）impala
	cloudera ： cdh（建议大家在生产上使用的hadoop系列版本）、cm
	sql：自己的守护进程执行的，非mr
	metastore

3）presto
	facebook
	京东
	sql

4）drill
	sql
	访问：hdfs、rdbms、json、hbase、mongodb、s3、hive

5）Spark SQL
	sql
	dataframe/dataset api
	metastore
	访问：hdfs、rdbms、json、hbase、mongodb、s3、hive  ==> 外部数据源



Spark SQL is Apache Spark's module for working with structured data. 

有见到SQL字样吗？
Spark SQL它不仅仅有访问或者操作SQL的功能，还提供了其他的非常丰富的操作：外部数据源、优化

Spark SQL概述小结：
1）Spark SQL的应用并不局限于SQL；
2）访问hive、json、parquet等文件的数据；
3）SQL只是Spark SQL的一个功能而已；
===> Spark SQL这个名字起的并不恰当
4）Spark SQL提供了SQL的api、DataFrame和Dataset的API；


```

## note5

```
提交Spark Application到环境中运行
spark-submit \
--name SQLContextApp \
--class com.imooc.spark.SQLContextApp \
--master local[2] \
/home/hadoop/lib/sql-1.0.jar \
/home/hadoop/app/spark-2.1.0-bin-2.6.0-cdh5.7.0/examples/src/main/resources/people.json


注意：
1）To use a HiveContext, you do not need to have an existing Hive setup
2）hive-site.xml



create table t(key string, value string);
explain extended select a.key*(2+3), b.value from  t a join t b on a.key = b.key and a.key > 3;

== Parsed Logical Plan ==
'Project [unresolvedalias(('a.key * (2 + 3)), None), 'b.value]
+- 'Join Inner, (('a.key = 'b.key) && ('a.key > 3))
   :- 'UnresolvedRelation `t`, a
   +- 'UnresolvedRelation `t`, b

== Analyzed Logical Plan ==
(CAST(key AS DOUBLE) * CAST((2 + 3) AS DOUBLE)): double, value: string
Project [(cast(key#321 as double) * cast((2 + 3) as double)) AS (CAST(key AS DOUBLE) * CAST((2 + 3) AS DOUBLE))#325, value#324]
+- Join Inner, ((key#321 = key#323) && (cast(key#321 as double) > cast(3 as double)))
   :- SubqueryAlias a
   :  +- MetastoreRelation default, t
   +- SubqueryAlias b
      +- MetastoreRelation default, t

== Optimized Logical Plan ==
Project [(cast(key#321 as double) * 5.0) AS (CAST(key AS DOUBLE) * CAST((2 + 3) AS DOUBLE))#325, value#324]
+- Join Inner, (key#321 = key#323)
   :- Project [key#321]
   :  +- Filter (isnotnull(key#321) && (cast(key#321 as double) > 3.0))
   :     +- MetastoreRelation default, t
   +- Filter (isnotnull(key#323) && (cast(key#323 as double) > 3.0))
      +- MetastoreRelation default, t

== Physical Plan ==
*Project [(cast(key#321 as double) * 5.0) AS (CAST(key AS DOUBLE) * CAST((2 + 3) AS DOUBLE))#325, value#324]
+- *SortMergeJoin [key#321], [key#323], Inner
   :- *Sort [key#321 ASC NULLS FIRST], false, 0
   :  +- Exchange hashpartitioning(key#321, 200)
   :     +- *Filter (isnotnull(key#321) && (cast(key#321 as double) > 3.0))
   :        +- HiveTableScan [key#321], MetastoreRelation default, t
   +- *Sort [key#323 ASC NULLS FIRST], false, 0
      +- Exchange hashpartitioning(key#323, 200)
         +- *Filter (isnotnull(key#323) && (cast(key#323 as double) > 3.0))
            +- HiveTableScan [key#323, value#324], MetastoreRelation default, t




thriftserver/beeline的使用
1) 启动thriftserver: 默认端口是10000 ，可以修改
2）启动beeline
beeline -u jdbc:hive2://localhost:10000 -n hadoop


修改thriftserver启动占用的默认端口号：
./start-thriftserver.sh  \
--master local[2] \
--jars ~/software/mysql-connector-java-5.1.27-bin.jar  \
--hiveconf hive.server2.thrift.port=14000 

beeline -u jdbc:hive2://localhost:14000 -n hadoop


thriftserver和普通的spark-shell/spark-sql有什么区别？
1）spark-shell、spark-sql都是一个spark  application；
2）thriftserver， 不管你启动多少个客户端(beeline/code)，永远都是一个spark application
	解决了一个数据共享的问题，多个客户端可以共享数据；


注意事项：在使用jdbc开发时，一定要先启动thriftserver
Exception in thread "main" java.sql.SQLException: 
Could not open client transport with JDBC Uri: jdbc:hive2://hadoop001:14000: 
java.net.ConnectException: Connection refused
```

## note6

```
DataFrame它不是Spark SQL提出的，而是早起在R、Pandas语言就已经有了的。


A Dataset is a distributed collection of data：分布式的数据集

A DataFrame is a Dataset organized into named columns. 
以列（列名、列的类型、列值）的形式构成的分布式数据集，按照列赋予不同的名称

student
id:int
name:string
city:string


It is conceptually equivalent to a table in a relational database 
or a data frame in R/Python


RDD： 
	java/scala  ==> jvm
	python ==> python runtime


DataFrame:
	java/scala/python ==> Logic Plan


DataFrame和RDD互操作的两种方式：
1）反射：case class   前提：事先需要知道你的字段、字段类型    
2）编程：Row          如果第一种情况不能满足你的要求（事先不知道列）
3) 选型：优先考虑第一种



val rdd = spark.sparkContext.textFile("file:///home/hadoop/data/student.data")



DataFrame = Dataset[Row]
Dataset：强类型  typed  case class
DataFrame：弱类型   Row


SQL: 
	seletc name from person;  compile  ok, result no

DF:
	df.select("name")  compile no
	df.select("nname")  compile ok  

DS:
	ds.map(line => line.itemid)  compile no


```

## note7

```
用户：
	方便快速从不同的数据源（json、parquet、rdbms），经过混合处理（json join parquet），
	再将处理结果以特定的格式（json、parquet）写回到指定的系统（HDFS、S3）上去


Spark SQL 1.2 ==> 外部数据源API


外部数据源的目的
1）开发人员：是否需要把代码合并到spark中？？？？
	weibo
	--jars 

2）用户
	读：spark.read.format(format)  
		format
			build-in: json parquet jdbc  csv(2+)
			packages: 外部的 并不是spark内置   https://spark-packages.org/
	写：people.write.format("parquet").save("path")		





处理parquet数据


RuntimeException: file:/home/hadoop/app/spark-2.1.0-bin-2.6.0-cdh5.7.0/examples/src/main/resources/people.json is not a Parquet file

  val DEFAULT_DATA_SOURCE_NAME = SQLConfigBuilder("spark.sql.sources.default")
    .doc("The default data source to use in input/output.")
    .stringConf
    .createWithDefault("parquet")

#注意USING的用法
CREATE TEMPORARY VIEW parquetTable
USING org.apache.spark.sql.parquet
OPTIONS (
  path "/home/hadoop/app/spark-2.1.0-bin-2.6.0-cdh5.7.0/examples/src/main/resources/users.parquet"
)

SELECT * FROM parquetTable


spark.sql("select deptno, count(1) as mount from emp where group by deptno").filter("deptno is not null").write.saveAsTable("hive_table_1")

org.apache.spark.sql.AnalysisException: Attribute name "count(1)" contains invalid character(s) among " ,;{}()\n\t=". Please use alias to rename it.;

spark.sqlContext.setConf("spark.sql.shuffle.partitions","10")

在生产环境中一定要注意设置spark.sql.shuffle.partitions，默认是200




操作MySQL的数据:
spark.read.format("jdbc").option("url", "jdbc:mysql://localhost:3306/hive").option("dbtable", "hive.TBLS").option("user", "root").option("password", "root").option("driver", "com.mysql.jdbc.Driver").load()

java.sql.SQLException: No suitable driver


import java.util.Properties
val connectionProperties = new Properties()
connectionProperties.put("user", "root")
connectionProperties.put("password", "root")
connectionProperties.put("driver", "com.mysql.jdbc.Driver")

val jdbcDF2 = spark.read.jdbc("jdbc:mysql://localhost:3306", "hive.TBLS", connectionProperties)


CREATE TEMPORARY VIEW jdbcTable
USING org.apache.spark.sql.jdbc
OPTIONS (
  url "jdbc:mysql://localhost:3306",
  dbtable "hive.TBLS",
  user 'root',
  password 'root',
  driver 'com.mysql.jdbc.Driver'
)


外部数据源综合案例
create database spark;
use spark;

CREATE TABLE DEPT(
DEPTNO int(2) PRIMARY KEY,
DNAME VARCHAR(14) ,
LOC VARCHAR(13) ) ;

INSERT INTO DEPT VALUES(10,'ACCOUNTING','NEW YORK');
INSERT INTO DEPT VALUES(20,'RESEARCH','DALLAS');
INSERT INTO DEPT VALUES(30,'SALES','CHICAGO');
INSERT INTO DEPT VALUES(40,'OPERATIONS','BOSTON');

```

## note8

```
用户行为日志：用户每次访问网站时所有的行为数据（访问、浏览、搜索、点击...）
	用户行为轨迹、流量日志


日志数据内容：
1）访问的系统属性： 操作系统、浏览器等等
2）访问特征：点击的url、从哪个url跳转过来的(referer)、页面上的停留时间等
3）访问信息：session_id、访问ip(访问城市)等

2013-05-19 13:00:00     http://www.taobao.com/17/?tracker_u=1624169&type=1      B58W48U4WKZCJ5D1T3Z9ZY88RU7QA7B1        http://hao.360.cn/      1.196.34.243   


数据处理流程
1）数据采集
	Flume： web日志写入到HDFS

2）数据清洗
	脏数据
	Spark、Hive、MapReduce 或者是其他的一些分布式计算框架  
	清洗完之后的数据可以存放在HDFS(Hive/Spark SQL)

3）数据处理
	按照我们的需要进行相应业务的统计和分析
	Spark、Hive、MapReduce 或者是其他的一些分布式计算框架

4）处理结果入库
	结果可以存放到RDBMS、NoSQL

5）数据的可视化
	通过图形化展示的方式展现出来：饼图、柱状图、地图、折线图
	ECharts、HUE、Zeppelin


一般的日志处理方式，我们是需要进行分区的，
按照日志中的访问时间进行相应的分区，比如：d,h,m5(每5分钟一个分区)


输入：访问时间、访问URL、耗费的流量、访问IP地址信息
输出：URL、cmsType(video/article)、cmsId(编号)、流量、ip、城市信息、访问时间、天



使用github上已有的开源项目
1）git clone https://github.com/wzhe06/ipdatabase.git
2）编译下载的项目：mvn clean package -DskipTests
3）安装jar包到自己的maven仓库
mvn install:install-file -Dfile=/Users/rocky/source/ipdatabase/target/ipdatabase-1.0-SNAPSHOT.jar -DgroupId=com.ggstar -DartifactId=ipdatabase -Dversion=1.0 -Dpackaging=jar


java.io.FileNotFoundException: 
file:/Users/rocky/maven_repos/com/ggstar/ipdatabase/1.0/ipdatabase-1.0.jar!/ipRegion.xlsx (No such file or directory)


调优点：
1) 控制文件输出的大小： coalesce
2) 分区字段的数据类型调整：spark.sql.sources.partitionColumnTypeInference.enabled
3) 批量插入数据库数据，提交使用batch操作

create table day_video_access_topn_stat (
day varchar(8) not null,
cms_id bigint(10) not null,
times bigint(10) not null,
primary key (day, cms_id)
);


create table day_video_city_access_topn_stat (
day varchar(8) not null,
cms_id bigint(10) not null,
city varchar(20) not null,
times bigint(10) not null,
times_rank int not null,
primary key (day, cms_id, city)
);

create table day_video_traffics_topn_stat (
day varchar(8) not null,
cms_id bigint(10) not null,
traffics bigint(20) not null,
primary key (day, cms_id)
);


数据可视化：一副图片最伟大的价值莫过于它能够使得我们实际看到的比我们期望看到的内容更加丰富

常见的可视化框架
1）echarts
2）highcharts
3）D3.js
4）HUE 
5）Zeppelin

在Spark中，支持4种运行模式：
1）Local：开发时使用
2）Standalone： 是Spark自带的，如果一个集群是Standalone的话，那么就需要在多台机器上同时部署Spark环境
3）YARN：建议大家在生产上使用该模式，统一使用YARN进行整个集群作业(MR、Spark)的资源调度
4）Mesos

不管使用什么模式，Spark应用程序的代码是一模一样的，只需要在提交的时候通过--master参数来指定我们的运行模式即可

Client
	Driver运行在Client端(提交Spark作业的机器)
	Client会和请求到的Container进行通信来完成作业的调度和执行，Client是不能退出的
	日志信息会在控制台输出：便于我们测试

Cluster
	Driver运行在ApplicationMaster中
	Client只要提交完作业之后就可以关掉，因为作业已经在YARN上运行了
	日志是在终端看不到的，因为日志是在Driver上，只能通过yarn logs -applicationIdapplication_id


./bin/spark-submit \
--class org.apache.spark.examples.SparkPi \
--master yarn \
--executor-memory 1G \
--num-executors 1 \
/home/hadoop/app/spark-2.1.0-bin-2.6.0-cdh5.7.0/examples/jars/spark-examples_2.11-2.1.0.jar \
4


此处的yarn就是我们的yarn client模式
如果是yarn cluster模式的话，yarn-cluster


Exception in thread "main" java.lang.Exception: When running with master 'yarn' either HADOOP_CONF_DIR or YARN_CONF_DIR must be set in the environment.

如果想运行在YARN之上，那么就必须要设置HADOOP_CONF_DIR或者是YARN_CONF_DIR

1） export HADOOP_CONF_DIR=/home/hadoop/app/hadoop-2.6.0-cdh5.7.0/etc/hadoop
2) $SPARK_HOME/conf/spark-env.sh


./bin/spark-submit \
--class org.apache.spark.examples.SparkPi \
--master yarn-cluster \
--executor-memory 1G \
--num-executors 1 \
/home/hadoop/app/spark-2.1.0-bin-2.6.0-cdh5.7.0/examples/jars/spark-examples_2.11-2.1.0.jar \
4


yarn logs -applicationId application_1495632775836_0002



打包时要注意，pom.xml中需要添加如下plugin
<plugin>
    <artifactId>maven-assembly-plugin</artifactId>
    <configuration>
        <archive>
            <manifest>
                <mainClass></mainClass>
            </manifest>
        </archive>
        <descriptorRefs>
            <descriptorRef>jar-with-dependencies</descriptorRef>
        </descriptorRefs>
    </configuration>
</plugin>

mvn assembly:assembly



./bin/spark-submit \
--class com.imooc.log.SparkStatCleanJobYARN \
--name SparkStatCleanJobYARN \
--master yarn \
--executor-memory 1G \
--num-executors 1 \
--files /home/hadoop/lib/ipDatabase.csv,/home/hadoop/lib/ipRegion.xlsx \
/home/hadoop/lib/sql-1.0-jar-with-dependencies.jar \
hdfs://hadoop001:8020/imooc/input/* hdfs://hadoop001:8020/imooc/clean

注意：--files在spark中的使用

spark.read.format("parquet").load("/imooc/clean/day=20170511/part-00000-71d465d1-7338-4016-8d1a-729504a9f95e.snappy.parquet").show(false)


./bin/spark-submit \
--class com.imooc.log.TopNStatJobYARN \
--name TopNStatJobYARN \
--master yarn \
--executor-memory 1G \
--num-executors 1 \
/home/hadoop/lib/sql-1.0-jar-with-dependencies.jar \
hdfs://hadoop001:8020/imooc/clean 20170511 

存储格式的选择：http://www.infoq.com/cn/articles/bigdata-store-choose/
压缩格式的选择：https://www.ibm.com/developerworks/cn/opensource/os-cn-hadoop-compression-analysis/

调整并行度
./bin/spark-submit \
--class com.imooc.log.TopNStatJobYARN \
--name TopNStatJobYARN \
--master yarn \
--executor-memory 1G \
--num-executors 1 \
--conf spark.sql.shuffle.partitions=100 \
/home/hadoop/lib/sql-1.0-jar-with-dependencies.jar \
hdfs://hadoop001:8020/imooc/clean 20170511 
```

## note9

```
即席查询
普通查询

Load Data
1) RDD    DataFrame/Dataset
2) Local   Cloud(HDFS/S3)


将数据加载成RDD
val masterLog = sc.textFile("file:///home/hadoop/app/spark-2.1.0-bin-2.6.0-cdh5.7.0/logs/spark-hadoop-org.apache.spark.deploy.master.Master-1-hadoop001.out")
val workerLog = sc.textFile("file:///home/hadoop/app/spark-2.1.0-bin-2.6.0-cdh5.7.0/logs/spark-hadoop-org.apache.spark.deploy.worker.Worker-1-hadoop001.out")
val allLog = sc.textFile("file:///home/hadoop/app/spark-2.1.0-bin-2.6.0-cdh5.7.0/logs/*out*")

masterLog.count
workerLog.count
allLog.count

存在的问题：使用使用SQL进行查询呢？

import org.apache.spark.sql.Row
val masterRDD = masterLog.map(x => Row(x))
import org.apache.spark.sql.types._
val schemaString = "line"

val fields = schemaString.split(" ").map(fieldName => StructField(fieldName, StringType, nullable = true))
val schema = StructType(fields)

val masterDF = spark.createDataFrame(masterRDD, schema)
masterDF.show


JSON/Parquet
val usersDF = spark.read.format("parquet").load("file:///home/hadoop/app/spark-2.1.0-bin-2.6.0-cdh5.7.0/examples/src/main/resources/users.parquet")
usersDF.show


spark.sql("select * from  parquet.`file:///home/hadoop/app/spark-2.1.0-bin-2.6.0-cdh5.7.0/examples/src/main/resources/users.parquet`").show

Drill 大数据处理框架


从Cloud读取数据: HDFS/S3
val hdfsRDD = sc.textFile("hdfs://path/file")
val s3RDD = sc.textFile("s3a://bucket/object")
	s3a/s3n

spark.read.format("text").load("hdfs://path/file")
spark.read.format("text").load("s3a://bucket/object")





val df=spark.read.format("json").load("file:///home/hadoop/app/spark-2.1.0-bin-2.6.0-cdh5.7.0/examples/src/main/resources/people.json")

df.show


TPC-DS


spark-packages.org
```

## 章节代码总结

```
package com.imooc.spark

import org.apache.spark.sql.SparkSession

/**
 * DataFrame中的操作操作
 */
object DataFrameCase {

  def main(args: Array[String]) {
    val spark = SparkSession.builder().appName("DataFrameRDDApp").master("local[2]").getOrCreate()

    // RDD ==> DataFrame
    val rdd = spark.sparkContext.textFile("file:///Users/rocky/data/student.data")

    //注意：需要导入隐式转换
    import spark.implicits._
    val studentDF = rdd.map(_.split("\\|")).map(line => Student(line(0).toInt, line(1), line(2), line(3))).toDF()

    //show默认只显示前20条
    studentDF.show
    studentDF.show(30)
    studentDF.show(30, false)

    studentDF.take(10)
    studentDF.first()
    studentDF.head(3)


    studentDF.select("email").show(30,false)


    studentDF.filter("name=''").show
    studentDF.filter("name='' OR name='NULL'").show


    //name以M开头的人
    studentDF.filter("SUBSTR(name,0,1)='M'").show

    studentDF.sort(studentDF("name")).show
    studentDF.sort(studentDF("name").desc).show

    studentDF.sort("name","id").show
    studentDF.sort(studentDF("name").asc, studentDF("id").desc).show

    studentDF.select(studentDF("name").as("student_name")).show


    val studentDF2 = rdd.map(_.split("\\|")).map(line => Student(line(0).toInt, line(1), line(2), line(3))).toDF()

    studentDF.join(studentDF2, studentDF.col("id") === studentDF2.col("id")).show

    spark.stop()
  }

  case class Student(id: Int, name: String, phone: String, email: String)

}
-----------------------------------------------------------------------
package com.imooc.spark

import org.apache.spark.sql.SparkSession

/**
 * DataFrame API基本操作
 */
object DataFrameApp {

  def main(args: Array[String]) {

    val spark = SparkSession.builder().appName("DataFrameApp").master("local[2]").getOrCreate()

    // 将json文件加载成一个dataframe
    val peopleDF = spark.read.format("json").load("file:///Users/rocky/data/people.json")

    // 输出dataframe对应的schema信息
    peopleDF.printSchema()

    // 输出数据集的前20条记录
    peopleDF.show()

    //查询某列所有的数据： select name from table
    peopleDF.select("name").show()

    // 查询某几列所有的数据，并对列进行计算： select name, age+10 as age2 from table
    peopleDF.select(peopleDF.col("name"), (peopleDF.col("age") + 10).as("age2")).show()

    //根据某一列的值进行过滤： select * from table where age>19
    peopleDF.filter(peopleDF.col("age") > 19).show()

    //根据某一列进行分组，然后再进行聚合操作： select age,count(1) from table group by age
    peopleDF.groupBy("age").count().show()

    spark.stop()
  }

}
-----------------------------------------------------------------------
package com.imooc.spark

import org.apache.spark.sql.types.{StringType, IntegerType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}

/**
 * DataFrame和RDD的互操作
 */
object DataFrameRDDApp {

  def main(args: Array[String]) {

    val spark = SparkSession.builder().appName("DataFrameRDDApp").master("local[2]").getOrCreate()

    //inferReflection(spark)

    program(spark)

    spark.stop()
  }

  def program(spark: SparkSession): Unit = {
    // RDD ==> DataFrame
    val rdd = spark.sparkContext.textFile("file:///Users/rocky/data/infos.txt")

    val infoRDD = rdd.map(_.split(",")).map(line => Row(line(0).toInt, line(1), line(2).toInt))

    val structType = StructType(Array(StructField("id", IntegerType, true),
      StructField("name", StringType, true),
      StructField("age", IntegerType, true)))

    val infoDF = spark.createDataFrame(infoRDD,structType)
    infoDF.printSchema()
    infoDF.show()


    //通过df的api进行操作
    infoDF.filter(infoDF.col("age") > 30).show

    //通过sql的方式进行操作
    infoDF.createOrReplaceTempView("infos")
    spark.sql("select * from infos where age > 30").show()
  }

  def inferReflection(spark: SparkSession) {
    // RDD ==> DataFrame
    val rdd = spark.sparkContext.textFile("file:///Users/rocky/data/infos.txt")

    //注意：需要导入隐式转换
    import spark.implicits._
    val infoDF = rdd.map(_.split(",")).map(line => Info(line(0).toInt, line(1), line(2).toInt)).toDF()

    infoDF.show()

    infoDF.filter(infoDF.col("age") > 30).show

    infoDF.createOrReplaceTempView("infos")
    spark.sql("select * from infos where age > 30").show()
  }

  case class Info(id: Int, name: String, age: Int)

}
----------------------------------------------------------------------
package com.imooc.spark

import org.apache.spark.sql.SparkSession

/**
 * Dataset操作
 */
object DatasetApp {

  def main(args: Array[String]) {
    val spark = SparkSession.builder().appName("DatasetApp")
      .master("local[2]").getOrCreate()

    //注意：需要导入隐式转换
    import spark.implicits._

    val path = "file:///Users/rocky/data/sales.csv"

    //spark如何解析csv文件？
    val df = spark.read.option("header","true").option("inferSchema","true").csv(path)
    df.show

    val ds = df.as[Sales]
    ds.map(line => line.itemId).show


    spark.sql("seletc name from person").show

    //df.seletc("name")
    df.select("nname")

    ds.map(line => line.itemId)

    spark.stop()
  }

  case class Sales(transactionId:Int,customerId:Int,itemId:Int,amountPaid:Double)
}
-----------------------------------------------------------------------
package com.imooc.spark

import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.{SparkConf, SparkContext}

/**
 * HiveContext的使用
 * 使用时需要通过--jars 把mysql的驱动传递到classpath
 */
object HiveContextApp {

  def main(args: Array[String]) {
    //1)创建相应的Context
    val sparkConf = new SparkConf()

    //在测试或者生产中，AppName和Master我们是通过脚本进行指定
    //sparkConf.setAppName("HiveContextApp").setMaster("local[2]")

    val sc = new SparkContext(sparkConf)
    val hiveContext = new HiveContext(sc)

    //2)相关的处理:
    hiveContext.table("emp").show

    //3)关闭资源
    sc.stop()
  }

}
----------------------------------------------------------------------
package com.imooc.spark

import org.apache.spark.sql.SparkSession

/**
 * 使用外部数据源综合查询Hive和MySQL的表数据
 */
object HiveMySQLApp {

  def main(args: Array[String]) {
    val spark = SparkSession.builder().appName("HiveMySQLApp")
      .master("local[2]").getOrCreate()

    // 加载Hive表数据
    val hiveDF = spark.table("emp")

    // 加载MySQL表数据
    val mysqlDF = spark.read.format("jdbc").option("url", "jdbc:mysql://localhost:3306").option("dbtable", "spark.DEPT").option("user", "root").option("password", "root").option("driver", "com.mysql.jdbc.Driver").load()

    // JOIN
    val resultDF = hiveDF.join(mysqlDF, hiveDF.col("deptno") === mysqlDF.col("DEPTNO"))
    resultDF.show


    resultDF.select(hiveDF.col("empno"),hiveDF.col("ename"),
      mysqlDF.col("deptno"), mysqlDF.col("dname")).show

    spark.stop()
  }

}
-----------------------------------------------------------------------
package com.imooc.spark

import org.apache.spark.sql.SparkSession

/**
 * Parquet文件操作
 */
object ParquetApp {

  def main(args: Array[String]) {

    val spark = SparkSession.builder().appName("SparkSessionApp")
      .master("local[2]").getOrCreate()


    /**
     * spark.read.format("parquet").load 这是标准写法
     */
    val userDF = spark.read.format("parquet").load("file:///home/hadoop/app/spark-2.1.0-bin-2.6.0-cdh5.7.0/examples/src/main/resources/users.parquet")

    userDF.printSchema()
    userDF.show()

    userDF.select("name","favorite_color").show

    userDF.select("name","favorite_color").write.format("json").save("file:///home/hadoop/tmp/jsonout")

    spark.read.load("file:///home/hadoop/app/spark-2.1.0-bin-2.6.0-cdh5.7.0/examples/src/main/resources/users.parquet").show

    //会报错，因为sparksql默认处理的format就是parquet
    spark.read.load("file:///home/hadoop/app/spark-2.1.0-bin-2.6.0-cdh5.7.0/examples/src/main/resources/people.json").show

    spark.read.format("parquet").option("path","file:///home/hadoop/app/spark-2.1.0-bin-2.6.0-cdh5.7.0/examples/src/main/resources/users.parquet").load().show
    spark.stop()
  }

}
-----------------------------------------------------------------------
package com.imooc.spark

import org.apache.spark.sql.SparkSession

/**
 * Schema Infer
 */
object SchemaInferApp {

  def main(args: Array[String]) {

    val spark = SparkSession.builder().appName("SchemaInferApp").master("local[2]").getOrCreate()

    val df = spark.read.format("json").load("file:///Users/rocky/data/json_schema_infer.json")

    df.printSchema()

    df.show()

    spark.stop()
  }

}
-----------------------------------------------------------------------
package com.imooc.spark

import org.apache.spark.sql.SparkSession

/**
 * SparkSession的使用
 */
object SparkSessionApp {

  def main(args: Array[String]) {

    val spark = SparkSession.builder().appName("SparkSessionApp")
      .master("local[2]").getOrCreate()

    val people = spark.read.json("file:///Users/rocky/data/people.json")
    people.show()

    spark.stop()
  }
}
---------------------------------------------------------------------
package com.imooc.spark

import java.sql.DriverManager

/**
 *  通过JDBC的方式访问
 */
object SparkSQLThriftServerApp {

  def main(args: Array[String]) {

    Class.forName("org.apache.hive.jdbc.HiveDriver")

    val conn = DriverManager.getConnection("jdbc:hive2://hadoop001:14000","hadoop","")
    val pstmt = conn.prepareStatement("select empno, ename, sal from emp")
    val rs = pstmt.executeQuery()
    while (rs.next()) {
      println("empno:" + rs.getInt("empno") +
        " , ename:" + rs.getString("ename") +
        " , sal:" + rs.getDouble("sal"))

    }
    rs.close()
    pstmt.close()
    conn.close()
  }
}
----------------------------------------------------------------------
package com.imooc.spark

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

import org.apache.spark.SparkConf

/**
 * SQLContext的使用:
 * 注意：IDEA是在本地，而测试数据是在服务器上 ，能不能在本地进行开发测试的？
 *
 */
object SQLContextApp {

  def main(args: Array[String]) {

    val path = args(0)

    //1)创建相应的Context
    val sparkConf = new SparkConf()

    //在测试或者生产中，AppName和Master我们是通过脚本进行指定
    //sparkConf.setAppName("SQLContextApp").setMaster("local[2]")

    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)

    //2)相关的处理: json
    val people = sqlContext.read.format("json").load(path)
    people.printSchema()
    people.show()



    //3)关闭资源
    sc.stop()
  }

}

```

## 日志实战源码

```scala
package com.imooc.log

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{LongType, StringType, StructField, StructType}

/**
 * 访问日志转换(输入==>输出)工具类
 */
object AccessConvertUtil {

  //定义的输出的字段
  val struct = StructType(
    Array(
      StructField("url",StringType),
      StructField("cmsType",StringType),
      StructField("cmsId",LongType),
      StructField("traffic",LongType),
      StructField("ip",StringType),
      StructField("city",StringType),
      StructField("time",StringType),
      StructField("day",StringType)
    )
  )

  /**
   * 根据输入的每一行信息转换成输出的样式
   * @param log  输入的每一行记录信息
   */
  def parseLog(log:String) = {

    try{
      val splits = log.split("\t")

      val url = splits(1)
      val traffic = splits(2).toLong
      val ip = splits(3)

      val domain = "http://www.imooc.com/"
      val cms = url.substring(url.indexOf(domain) + domain.length)
      val cmsTypeId = cms.split("/")

      var cmsType = ""
      var cmsId = 0l
      if(cmsTypeId.length > 1) {
        cmsType = cmsTypeId(0)
        cmsId = cmsTypeId(1).toLong
      }

      val city = IpUtils.getCity(ip)
      val time = splits(0)
      val day = time.substring(0,10).replaceAll("-","")

      //这个row里面的字段要和struct中的字段对应上
      Row(url, cmsType, cmsId, traffic, ip, city, time, day)
    } catch {
      case e:Exception => Row(0)
    }
  }
}
----------------------------------------------------------------------
package com.imooc.log

import java.util.{Date, Locale}

import org.apache.commons.lang3.time.FastDateFormat

/**
 * 日期时间解析工具类:
 * 注意：SimpleDateFormat是线程不安全
 */
object DateUtils {

  //输入文件日期时间格式
  //10/Nov/2016:00:01:02 +0800
  val YYYYMMDDHHMM_TIME_FORMAT = FastDateFormat.getInstance("dd/MMM/yyyy:HH:mm:ss Z", Locale.ENGLISH)

  //目标日期格式
  val TARGET_FORMAT = FastDateFormat.getInstance("yyyy-MM-dd HH:mm:ss")


  /**
   * 获取时间：yyyy-MM-dd HH:mm:ss
   */
  def parse(time: String) = {
    TARGET_FORMAT.format(new Date(getTime(time)))
  }

  /**
   * 获取输入日志时间：long类型
   *
   * time: [10/Nov/2016:00:01:02 +0800]
   */
  def getTime(time: String) = {
    try {
      YYYYMMDDHHMM_TIME_FORMAT.parse(time.substring(time.indexOf("[") + 1,
        time.lastIndexOf("]"))).getTime
    } catch {
      case e: Exception => {
        0l
      }
    }
  }

  def main(args: Array[String]) {
    println(parse("[10/Nov/2016:00:01:02 +0800]"))
  }

}
-----------------------------------------
package com.imooc.log

case class DayCityVideoAccessStat(day:String, cmsId:Long, city:String,times:Long,timesRank:Int)
package com.imooc.log
-----------------------------------------
/**
 * 每天课程访问次数实体类
 */
case class DayVideoAccessStat(day: String, cmsId: Long, times: Long)
-----------------------------------------
package com.imooc.log

case class DayVideoTrafficsStat(day:String,cmsId:Long,traffics:Long)
----------------------------------------------------------------
package com.imooc.log

import com.ggstar.util.ip.IpHelper

/**
 * IP解析工具类
 */
object IpUtils {


  def getCity(ip:String) = {
    IpHelper.findRegionByIp(ip)
  }

  def main(args: Array[String]) {
    println(getCity("218.75.35.226"))
  }

}
-------------------------------------------------------------------
package com.imooc.log

import java.sql.{Connection, PreparedStatement, DriverManager}

/**
 * MySQL操作工具类
 */
object MySQLUtils {

  /**
   * 获取数据库连接
   */
  def getConnection() = {
    DriverManager.getConnection("jdbc:mysql://localhost:3306/imooc_project?user=root&password=root")
  }

  /**
   * 释放数据库连接等资源
   * @param connection
   * @param pstmt
   */
  def release(connection: Connection, pstmt: PreparedStatement): Unit = {
    try {
      if (pstmt != null) {
        pstmt.close()
      }
    } catch {
      case e: Exception => e.printStackTrace()
    } finally {
      if (connection != null) {
        connection.close()
      }
    }
  }

  def main(args: Array[String]) {
    println(getConnection())
  }

}
---------------------------------------------------------------
package com.imooc.log

import org.apache.spark.sql.{SaveMode, SparkSession}

/**
 * 使用Spark完成我们的数据清洗操作
 */
object SparkStatCleanJob {

  def main(args: Array[String]) {
    val spark = SparkSession.builder().appName("SparkStatCleanJob")
      .config("spark.sql.parquet.compression.codec","gzip")
      .master("local[2]").getOrCreate()

    val accessRDD = spark.sparkContext.textFile("/Users/rocky/data/imooc/access.log")

    //accessRDD.take(10).foreach(println)

    //RDD ==> DF
    val accessDF = spark.createDataFrame(accessRDD.map(x => AccessConvertUtil.parseLog(x)),
      AccessConvertUtil.struct)

//    accessDF.printSchema()
//    accessDF.show(false)

    accessDF.coalesce(1).write.format("parquet").mode(SaveMode.Overwrite)
      .partitionBy("day").save("/Users/rocky/data/imooc/clean2")

    spark.stop
  }
}
----------------------------------------------------------------------
package com.imooc.log

import org.apache.spark.sql.{SaveMode, SparkSession}

/**
 * 使用Spark完成我们的数据清洗操作：运行在YARN之上
 */
object SparkStatCleanJobYARN {

  def main(args: Array[String]) {

    if(args.length !=2) {
      println("Usage: SparkStatCleanJobYARN <inputPath> <outputPath>")
      System.exit(1)
    }

    val Array(inputPath, outputPath) = args

    val spark = SparkSession.builder().getOrCreate()

    val accessRDD = spark.sparkContext.textFile(inputPath)

    //RDD ==> DF
    val accessDF = spark.createDataFrame(accessRDD.map(x => AccessConvertUtil.parseLog(x)),
      AccessConvertUtil.struct)

    accessDF.coalesce(1).write.format("parquet").mode(SaveMode.Overwrite)
      .partitionBy("day").save(outputPath)

    spark.stop
  }
}
----------------------------------------------------------------------
package com.imooc.log

import org.apache.spark.sql.SparkSession

/**
 * 第一步清洗：抽取出我们所需要的指定列的数据
 */
object SparkStatFormatJob {

  def main(args: Array[String]) {

    val spark = SparkSession.builder().appName("SparkStatFormatJob")
      .master("local[2]").getOrCreate()

    val acccess = spark.sparkContext.textFile("file:///Users/rocky/data/imooc/10000_access.log")

    //acccess.take(10).foreach(println)

    acccess.map(line => {
      val splits = line.split(" ")
      val ip = splits(0)

      /**
       * 原始日志的第三个和第四个字段拼接起来就是完整的访问时间：
       * [10/Nov/2016:00:01:02 +0800] ==> yyyy-MM-dd HH:mm:ss
       */
      val time = splits(3) + " " + splits(4)
      val url = splits(11).replaceAll("\"","")
      val traffic = splits(9)
//      (ip, DateUtils.parse(time), url, traffic)
      DateUtils.parse(time) + "\t" + url + "\t" + traffic + "\t" + ip
    }).saveAsTextFile("file:///Users/rocky/data/imooc/output/")

    spark.stop()
  }

}
----------------------------------------------------------------
package com.imooc.log

import java.sql.{PreparedStatement, Connection}

import scala.collection.mutable.ListBuffer

/**
 * 各个维度统计的DAO操作
 */
object StatDAO {


  /**
   * 批量保存DayVideoAccessStat到数据库
   */
  def insertDayVideoAccessTopN(list: ListBuffer[DayVideoAccessStat]): Unit = {

    var connection: Connection = null
    var pstmt: PreparedStatement = null

    try {
      connection = MySQLUtils.getConnection()

      connection.setAutoCommit(false) //设置手动提交

      val sql = "insert into day_video_access_topn_stat(day,cms_id,times) values (?,?,?) "
      pstmt = connection.prepareStatement(sql)

      for (ele <- list) {
        pstmt.setString(1, ele.day)
        pstmt.setLong(2, ele.cmsId)
        pstmt.setLong(3, ele.times)

        pstmt.addBatch()
      }

      pstmt.executeBatch() // 执行批量处理
      connection.commit() //手工提交
    } catch {
      case e: Exception => e.printStackTrace()
    } finally {
      MySQLUtils.release(connection, pstmt)
    }
  }


  /**
   * 批量保存DayCityVideoAccessStat到数据库
   */
  def insertDayCityVideoAccessTopN(list: ListBuffer[DayCityVideoAccessStat]): Unit = {

    var connection: Connection = null
    var pstmt: PreparedStatement = null

    try {
      connection = MySQLUtils.getConnection()

      connection.setAutoCommit(false) //设置手动提交

      val sql = "insert into day_video_city_access_topn_stat(day,cms_id,city,times,times_rank) values (?,?,?,?,?) "
      pstmt = connection.prepareStatement(sql)

      for (ele <- list) {
        pstmt.setString(1, ele.day)
        pstmt.setLong(2, ele.cmsId)
        pstmt.setString(3, ele.city)
        pstmt.setLong(4, ele.times)
        pstmt.setInt(5, ele.timesRank)
        pstmt.addBatch()
      }

      pstmt.executeBatch() // 执行批量处理
      connection.commit() //手工提交
    } catch {
      case e: Exception => e.printStackTrace()
    } finally {
      MySQLUtils.release(connection, pstmt)
    }
  }


  /**
   * 批量保存DayVideoTrafficsStat到数据库
   */
  def insertDayVideoTrafficsAccessTopN(list: ListBuffer[DayVideoTrafficsStat]): Unit = {

    var connection: Connection = null
    var pstmt: PreparedStatement = null

    try {
      connection = MySQLUtils.getConnection()

      connection.setAutoCommit(false) //设置手动提交

      val sql = "insert into day_video_traffics_topn_stat(day,cms_id,traffics) values (?,?,?) "
      pstmt = connection.prepareStatement(sql)

      for (ele <- list) {
        pstmt.setString(1, ele.day)
        pstmt.setLong(2, ele.cmsId)
        pstmt.setLong(3, ele.traffics)
        pstmt.addBatch()
      }

      pstmt.executeBatch() // 执行批量处理
      connection.commit() //手工提交
    } catch {
      case e: Exception => e.printStackTrace()
    } finally {
      MySQLUtils.release(connection, pstmt)
    }
  }


  /**
   * 删除指定日期的数据
   */
  def deleteData(day: String): Unit = {

    val tables = Array("day_video_access_topn_stat",
      "day_video_city_access_topn_stat",
      "day_video_traffics_topn_stat")

    var connection:Connection = null
    var pstmt:PreparedStatement = null

    try{
      connection = MySQLUtils.getConnection()

      for(table <- tables) {
        // delete from table ....
        val deleteSQL = s"delete from $table where day = ?"
        pstmt = connection.prepareStatement(deleteSQL)
        pstmt.setString(1, day)
        pstmt.executeUpdate()
      }
    }catch {
      case e:Exception => e.printStackTrace()
    } finally {
      MySQLUtils.release(connection, pstmt)
    }


  }
}
--------------------------------------------------------------------
package com.imooc.log

import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable.ListBuffer

/**
 * TopN统计Spark作业
 */
object TopNStatJob {

  def main(args: Array[String]) {
    val spark = SparkSession.builder().appName("TopNStatJob")
      .config("spark.sql.sources.partitionColumnTypeInference.enabled","false")
      .master("local[2]").getOrCreate()


    val accessDF = spark.read.format("parquet").load("/Users/rocky/data/imooc/clean")

//    accessDF.printSchema()
//    accessDF.show(false)

    val day = "20170511"

    StatDAO.deleteData(day)

    //最受欢迎的TopN课程
    videoAccessTopNStat(spark, accessDF, day)

    //按照地市进行统计TopN课程
    cityAccessTopNStat(spark, accessDF, day)

    //按照流量进行统计
    videoTrafficsTopNStat(spark, accessDF, day)

    spark.stop()
  }

  /**
   * 按照流量进行统计
   */
  def videoTrafficsTopNStat(spark: SparkSession, accessDF:DataFrame, day:String): Unit = {
    import spark.implicits._

    val cityAccessTopNDF = accessDF.filter($"day" === day && $"cmsType" === "video")
    .groupBy("day","cmsId").agg(sum("traffic").as("traffics"))
    .orderBy($"traffics".desc)
    //.show(false)

    /**
     * 将统计结果写入到MySQL中
     */
    try {
      cityAccessTopNDF.foreachPartition(partitionOfRecords => {
        val list = new ListBuffer[DayVideoTrafficsStat]

        partitionOfRecords.foreach(info => {
          val day = info.getAs[String]("day")
          val cmsId = info.getAs[Long]("cmsId")
          val traffics = info.getAs[Long]("traffics")
          list.append(DayVideoTrafficsStat(day, cmsId,traffics))
        })

        StatDAO.insertDayVideoTrafficsAccessTopN(list)
      })
    } catch {
      case e:Exception => e.printStackTrace()
    }

  }

  /**
   * 按照地市进行统计TopN课程
   */
  def cityAccessTopNStat(spark: SparkSession, accessDF:DataFrame, day:String): Unit = {
    import spark.implicits._

    val cityAccessTopNDF = accessDF.filter($"day" === day && $"cmsType" === "video")
    .groupBy("day","city","cmsId")
    .agg(count("cmsId").as("times"))

    //cityAccessTopNDF.show(false)

    //Window函数在Spark SQL的使用

    val top3DF = cityAccessTopNDF.select(
      cityAccessTopNDF("day"),
      cityAccessTopNDF("city"),
      cityAccessTopNDF("cmsId"),
      cityAccessTopNDF("times"),
      row_number().over(Window.partitionBy(cityAccessTopNDF("city"))
      .orderBy(cityAccessTopNDF("times").desc)
      ).as("times_rank")
    ).filter("times_rank <=3") //.show(false)  //Top3


    /**
     * 将统计结果写入到MySQL中
     */
    try {
      top3DF.foreachPartition(partitionOfRecords => {
        val list = new ListBuffer[DayCityVideoAccessStat]

        partitionOfRecords.foreach(info => {
          val day = info.getAs[String]("day")
          val cmsId = info.getAs[Long]("cmsId")
          val city = info.getAs[String]("city")
          val times = info.getAs[Long]("times")
          val timesRank = info.getAs[Int]("times_rank")
          list.append(DayCityVideoAccessStat(day, cmsId, city, times, timesRank))
        })

        StatDAO.insertDayCityVideoAccessTopN(list)
      })
    } catch {
      case e:Exception => e.printStackTrace()
    }

  }


    /**
   * 最受欢迎的TopN课程
   */
  def videoAccessTopNStat(spark: SparkSession, accessDF:DataFrame, day:String): Unit = {

    /**
     * 使用DataFrame的方式进行统计
     */
    import spark.implicits._

    val videoAccessTopNDF = accessDF.filter($"day" === day && $"cmsType" === "video")
    .groupBy("day","cmsId").agg(count("cmsId").as("times")).orderBy($"times".desc)

    videoAccessTopNDF.show(false)

    /**
     * 使用SQL的方式进行统计
     */
//    accessDF.createOrReplaceTempView("access_logs")
//    val videoAccessTopNDF = spark.sql("select day,cmsId, count(1) as times from access_logs " +
//      "where day='20170511' and cmsType='video' " +
//      "group by day,cmsId order by times desc")
//
//    videoAccessTopNDF.show(false)

    /**
     * 将统计结果写入到MySQL中
     */
    try {
      videoAccessTopNDF.foreachPartition(partitionOfRecords => {
        val list = new ListBuffer[DayVideoAccessStat]

        partitionOfRecords.foreach(info => {
          val day = info.getAs[String]("day")
          val cmsId = info.getAs[Long]("cmsId")
          val times = info.getAs[Long]("times")

          /**
           * 不建议大家在此处进行数据库的数据插入
           */

          list.append(DayVideoAccessStat(day, cmsId, times))
        })

        StatDAO.insertDayVideoAccessTopN(list)
      })
    } catch {
      case e:Exception => e.printStackTrace()
    }

  }

}
---------------------------------------------------------------------
package com.imooc.log

import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable.ListBuffer

/**
 * TopN统计Spark作业：复用已有的数据
 */
object TopNStatJob2 {

  def main(args: Array[String]) {
    val spark = SparkSession.builder().appName("TopNStatJob")
      .config("spark.sql.sources.partitionColumnTypeInference.enabled","false")
      .master("local[2]").getOrCreate()


    val accessDF = spark.read.format("parquet").load("/Users/rocky/data/imooc/clean")

//    accessDF.printSchema()
//    accessDF.show(false)

    val day = "20170511"

    import spark.implicits._
    val commonDF = accessDF.filter($"day" === day && $"cmsType" === "video")

    commonDF.cache()

    StatDAO.deleteData(day)

    //最受欢迎的TopN课程
    videoAccessTopNStat(spark, commonDF)

    //按照地市进行统计TopN课程
    cityAccessTopNStat(spark, commonDF)

    //按照流量进行统计
    videoTrafficsTopNStat(spark, commonDF)

    commonDF.unpersist(true)

    spark.stop()
  }

  /**
   * 按照流量进行统计
   */
  def videoTrafficsTopNStat(spark: SparkSession, commonDF:DataFrame): Unit = {
    import spark.implicits._

    val cityAccessTopNDF = commonDF.groupBy("day","cmsId")
      .agg(sum("traffic").as("traffics"))
    .orderBy($"traffics".desc)
    //.show(false)

    /**
     * 将统计结果写入到MySQL中
     */
    try {
      cityAccessTopNDF.foreachPartition(partitionOfRecords => {
        val list = new ListBuffer[DayVideoTrafficsStat]

        partitionOfRecords.foreach(info => {
          val day = info.getAs[String]("day")
          val cmsId = info.getAs[Long]("cmsId")
          val traffics = info.getAs[Long]("traffics")
          list.append(DayVideoTrafficsStat(day, cmsId,traffics))
        })

        StatDAO.insertDayVideoTrafficsAccessTopN(list)
      })
    } catch {
      case e:Exception => e.printStackTrace()
    }

  }

  /**
   * 按照地市进行统计TopN课程
   */
  def cityAccessTopNStat(spark: SparkSession, commonDF:DataFrame): Unit = {

    val cityAccessTopNDF = commonDF
    .groupBy("day","city","cmsId")
    .agg(count("cmsId").as("times"))

    //cityAccessTopNDF.show(false)

    //Window函数在Spark SQL的使用

    val top3DF = cityAccessTopNDF.select(
      cityAccessTopNDF("day"),
      cityAccessTopNDF("city"),
      cityAccessTopNDF("cmsId"),
      cityAccessTopNDF("times"),
      row_number().over(Window.partitionBy(cityAccessTopNDF("city"))
      .orderBy(cityAccessTopNDF("times").desc)
      ).as("times_rank")
    ).filter("times_rank <=3") //.show(false)  //Top3


    /**
     * 将统计结果写入到MySQL中
     */
    try {
      top3DF.foreachPartition(partitionOfRecords => {
        val list = new ListBuffer[DayCityVideoAccessStat]

        partitionOfRecords.foreach(info => {
          val day = info.getAs[String]("day")
          val cmsId = info.getAs[Long]("cmsId")
          val city = info.getAs[String]("city")
          val times = info.getAs[Long]("times")
          val timesRank = info.getAs[Int]("times_rank")
          list.append(DayCityVideoAccessStat(day, cmsId, city, times, timesRank))
        })

        StatDAO.insertDayCityVideoAccessTopN(list)
      })
    } catch {
      case e:Exception => e.printStackTrace()
    }

  }


    /**
   * 最受欢迎的TopN课程
   */
  def videoAccessTopNStat(spark: SparkSession, commonDF:DataFrame): Unit = {

    /**
     * 使用DataFrame的方式进行统计
     */
    import spark.implicits._

    val videoAccessTopNDF = commonDF
    .groupBy("day","cmsId").agg(count("cmsId").as("times")).orderBy($"times".desc)

    videoAccessTopNDF.show(false)

    /**
     * 使用SQL的方式进行统计
     */
//    accessDF.createOrReplaceTempView("access_logs")
//    val videoAccessTopNDF = spark.sql("select day,cmsId, count(1) as times from access_logs " +
//      "where day='20170511' and cmsType='video' " +
//      "group by day,cmsId order by times desc")
//
//    videoAccessTopNDF.show(false)

    /**
     * 将统计结果写入到MySQL中
     */
    try {
      videoAccessTopNDF.foreachPartition(partitionOfRecords => {
        val list = new ListBuffer[DayVideoAccessStat]

        partitionOfRecords.foreach(info => {
          val day = info.getAs[String]("day")
          val cmsId = info.getAs[Long]("cmsId")
          val times = info.getAs[Long]("times")

          /**
           * 不建议大家在此处进行数据库的数据插入
           */

          list.append(DayVideoAccessStat(day, cmsId, times))
        })

        StatDAO.insertDayVideoAccessTopN(list)
      })
    } catch {
      case e:Exception => e.printStackTrace()
    }

  }

}
-----------------------------------------------------------------------
package com.imooc.log

import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable.ListBuffer

/**
 * TopN统计Spark作业:运行在YARN之上
 */
object TopNStatJobYARN {

  def main(args: Array[String]) {

    if(args.length !=2) {
      println("Usage: TopNStatJobYARN <inputPath> <day>")
      System.exit(1)
    }

    val Array(inputPath, day) = args
    val spark = SparkSession.builder()
      .config("spark.sql.sources.partitionColumnTypeInference.enabled","false")
      .getOrCreate()


    val accessDF = spark.read.format("parquet").load(inputPath)

    StatDAO.deleteData(day)

    //最受欢迎的TopN课程
    videoAccessTopNStat(spark, accessDF, day)

    //按照地市进行统计TopN课程
    cityAccessTopNStat(spark, accessDF, day)

    //按照流量进行统计
    videoTrafficsTopNStat(spark, accessDF, day)

    spark.stop()
  }

  /**
   * 按照流量进行统计
   */
  def videoTrafficsTopNStat(spark: SparkSession, accessDF:DataFrame, day:String): Unit = {
    import spark.implicits._

    val cityAccessTopNDF = accessDF.filter($"day" === day && $"cmsType" === "video")
    .groupBy("day","cmsId").agg(sum("traffic").as("traffics"))
    .orderBy($"traffics".desc)
    //.show(false)

    /**
     * 将统计结果写入到MySQL中
     */
    try {
      cityAccessTopNDF.foreachPartition(partitionOfRecords => {
        val list = new ListBuffer[DayVideoTrafficsStat]

        partitionOfRecords.foreach(info => {
          val day = info.getAs[String]("day")
          val cmsId = info.getAs[Long]("cmsId")
          val traffics = info.getAs[Long]("traffics")
          list.append(DayVideoTrafficsStat(day, cmsId,traffics))
        })

        StatDAO.insertDayVideoTrafficsAccessTopN(list)
      })
    } catch {
      case e:Exception => e.printStackTrace()
    }

  }

  /**
   * 按照地市进行统计TopN课程
   */
  def cityAccessTopNStat(spark: SparkSession, accessDF:DataFrame, day:String): Unit = {
    import spark.implicits._

    val cityAccessTopNDF = accessDF.filter($"day" === day && $"cmsType" === "video")
    .groupBy("day","city","cmsId")
    .agg(count("cmsId").as("times"))

    //cityAccessTopNDF.show(false)

    //Window函数在Spark SQL的使用

    val top3DF = cityAccessTopNDF.select(
      cityAccessTopNDF("day"),
      cityAccessTopNDF("city"),
      cityAccessTopNDF("cmsId"),
      cityAccessTopNDF("times"),
      row_number().over(Window.partitionBy(cityAccessTopNDF("city"))
      .orderBy(cityAccessTopNDF("times").desc)
      ).as("times_rank")
    ).filter("times_rank <=3") //.show(false)  //Top3


    /**
     * 将统计结果写入到MySQL中
     */
    try {
      top3DF.foreachPartition(partitionOfRecords => {
        val list = new ListBuffer[DayCityVideoAccessStat]

        partitionOfRecords.foreach(info => {
          val day = info.getAs[String]("day")
          val cmsId = info.getAs[Long]("cmsId")
          val city = info.getAs[String]("city")
          val times = info.getAs[Long]("times")
          val timesRank = info.getAs[Int]("times_rank")
          list.append(DayCityVideoAccessStat(day, cmsId, city, times, timesRank))
        })

        StatDAO.insertDayCityVideoAccessTopN(list)
      })
    } catch {
      case e:Exception => e.printStackTrace()
    }

  }


    /**
   * 最受欢迎的TopN课程
   */
  def videoAccessTopNStat(spark: SparkSession, accessDF:DataFrame, day:String): Unit = {

    /**
     * 使用DataFrame的方式进行统计
     */
    import spark.implicits._

    val videoAccessTopNDF = accessDF.filter($"day" === day && $"cmsType" === "video")
    .groupBy("day","cmsId").agg(count("cmsId").as("times")).orderBy($"times".desc)

    videoAccessTopNDF.show(false)

    /**
     * 使用SQL的方式进行统计
     */
//    accessDF.createOrReplaceTempView("access_logs")
//    val videoAccessTopNDF = spark.sql("select day,cmsId, count(1) as times from access_logs " +
//      "where day='20170511' and cmsType='video' " +
//      "group by day,cmsId order by times desc")
//
//    videoAccessTopNDF.show(false)

    /**
     * 将统计结果写入到MySQL中
     */
    try {
      videoAccessTopNDF.foreachPartition(partitionOfRecords => {
        val list = new ListBuffer[DayVideoAccessStat]

        partitionOfRecords.foreach(info => {
          val day = info.getAs[String]("day")
          val cmsId = info.getAs[Long]("cmsId")
          val times = info.getAs[Long]("times")

          list.append(DayVideoAccessStat(day, cmsId, times))
        })

        StatDAO.insertDayVideoAccessTopN(list)
      })
    } catch {
      case e:Exception => e.printStackTrace()
    }

  }

}

```


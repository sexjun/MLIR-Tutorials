Transform： 方言内转化
Conversion: 方言间转化
Analysis： 分析pass


`pass` 其实在IREE项目可以发现
主要是在 `Transform` 任务中，用与方言内部的转化。 pass的声明都是在 `Passes.td`文件中完成的。


`Pattern` 主要是 `Conversion` 任务中， 用于方言间的转化。
<!-- `OpConversionPattern` -->


还需要了解 Analysis 的作用， 以及pass 的 `pipleline`




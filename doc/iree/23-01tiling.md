youtune 上看到的频道： https://www.youtube.com/watch?v=NydMjQyzaQQ&t=729
字幕存在同级路径，之后可以通过AI咨询。



- 是在哪个方言完成的tiling？

数据平铺（tiling）工作主要涉及多个 IREE 方言，具体如下：
Flow Dialect 层面
在此层面，主要任务是识别原子工作单元。对于数据平铺操作，不需要知道具体使用的布局，但需要知道特定张量将进行数据平铺，并且可能需要一种方法来区分不同的布局（如矩阵乘法中左右操作数可能有不同布局）。通过挂钩到现有的张量编码机制，利用设置和取消编码操作来标记数据平铺。这里使用的编码机制是基于上游稀疏方言工作中的张量编码，其目的是在不泄露架构细节的同时，将布局相关信息在程序中传递，并且尽可能晚地处理布局变化，以保留计算语义信息。例如，对于需要改变布局的张量值，添加编码表示其实际布局与默认不同，这种编码会在后续流程中传播并转换为实际的布局参数。
Stream Dialect 层面
负责调度由 Flow Dialect 划分的调度器（dispatchers）。在这个层面，虽然不涉及数据平铺操作的核心计算，但与数据分配等相关工作有关。例如，数据分配在这个层面发生，而数据平铺操作中的 pack 和 unpack 操作需要明确的内瓷砖大小，这些信息在 Stream Dialect 层面进行管理，确保为目标设备分配合适数量的内存，并且可以在运行时查询内存分配的填充情况。虽然 Stream Dialect 本身不直接处理数据平铺的核心计算，但它为后续在 Higher Dialect 中进行实际的计算操作提供了必要的数据准备和调度基础。
Higher Dialect 层面
在这个方言中，拥有关于目标架构的详细信息，能够根据这些信息将编码转换为实际的布局参数，包括外密集排列、内维度位置和内瓷砖大小等。这是数据平铺操作在计算层面得以实现的关键环节，因为在这里可以根据具体架构将之前标记的数据平铺信息转化为实际的计算指令和内存访问模式。例如，当处理带有编码的线性点积（linear dot magma）操作数时，可以将其转换为适合架构的线性通用（linear generic）操作，从而明确张量维度和布局，为代码生成提供所需的全部信息，使得计算操作能够以平铺的方式高效执行，实现数据平铺在性能优化方面的目标，如提高内存访问效率和向量化计算等。


- 在不同的方言中，数据平铺的具体实现方式有何不同？

Flow Dialect
布局标记与抽象表示
主要负责识别原子工作单元，在数据平铺中，不需要明确具体的布局细节，但要知道哪些张量将进行数据平铺，以及区分不同布局（如矩阵乘法左右操作数布局可能不同）。
通过挂钩到张量编码机制来标记数据平铺，利用设置和取消编码操作实现。例如，对于需改变布局的张量值，添加编码表明其实际布局与默认不同，这种编码基于上游稀疏方言工作，目的是在不泄露架构细节的情况下传递布局信息，且尽量晚处理布局变化以保留计算语义。
信息传递与初步处理
标记的编码信息会传播到后续方言。此层面不涉及复杂计算操作，重点在于为后续处理提供必要的布局信息标记，为整个数据平铺流程奠定基础，确保数据在后续处理中能依据合适的布局进行操作。
Stream Dialect
调度与内存管理
负责调度由 Flow Dialect 划分的调度器，虽然不直接进行数据平铺的核心计算，但与数据分配紧密相关。
数据分配在此发生，如为数据平铺操作中的 pack 和 unpack 操作确保合适内存分配，管理内瓷砖大小相关信息，确保为目标设备分配合适内存量，并可在运行时查询内存分配的填充情况，为数据在硬件上的实际处理做好准备工作，保障数据在不同设备上能按需求正确分配和访问。
Higher Dialect
架构相关的布局转换
依据目标架构将编码转换为实际布局参数，包括外密集排列、内维度位置和内瓷砖大小等。例如处理带有编码的线性点积操作数时，可将其转换为适合架构的线性通用操作，使张量维度和布局明确化。
计算指令生成
基于转换后的布局参数，生成代码并执行计算操作，实现数据平铺在性能优化方面的目标，如提高内存访问效率和向量化计算等，将数据平铺从抽象的布局标记和信息传递落实到实际的计算执行层面，完成整个数据平铺流程在不同架构上的适配和优化。
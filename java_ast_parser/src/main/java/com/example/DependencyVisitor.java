package com.example;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.concurrent.*;

import com.example.dependencies.*;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Modifier;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.Type;
import com.github.javaparser.ast.type.TypeParameter;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.resolution.UnsolvedSymbolException;
import com.github.javaparser.resolution.types.ResolvedType;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JarTypeSolver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;

public class DependencyVisitor extends VoidVisitorAdapter<NodeContext> {
    private static final Logger logger = LogManager.getLogger(DependencyVisitor.class);

    private NodeContext currentContext;
    private String currentClassName;
    private Stack<String> classStack = new Stack<>();
    private String currentFilePath;

    private DependencyVisitorConfig config;
    private Stack<BufferedWriter> dependencyWriterStack = new Stack<>();
    private BufferedWriter dependencyWriter;
    private BufferedWriter classInfoWriter;
    private File outputDir;
    private JavaParser javaParser;

    public DependencyVisitor(DependencyVisitorConfig config) {
        this.config = config;
        CombinedTypeSolver solver = new CombinedTypeSolver();
        if (config.isEnableReflectionTypeSolver()) {
            solver.add(new ReflectionTypeSolver());
        }
        if (config.isEnableJarTypeSolver()) {
            for (File jarFile : config.getJarFiles()) {
                try {
                    solver.add(new JarTypeSolver(jarFile));
                } catch (IOException e) {
                    logger.error("无法添加 JAR 文件: " + jarFile.getAbsolutePath());
                }
            }
        }
        for (File sourceDir : config.getSourceDirectories()) {
            solver.add(new JavaParserTypeSolver(sourceDir));
        }

        ParserConfiguration parserConfiguration = new ParserConfiguration()
                .setLanguageLevel(ParserConfiguration.LanguageLevel.JAVA_8)
                .setAttributeComments(false)
                .setSymbolResolver(new JavaSymbolSolver(solver));
        this.javaParser = new JavaParser(parserConfiguration);
    }

    public void process() {
        // 分批处理每个源目录
        for (File sourceDir : config.getDirectoriesToParse()) {
            processDirectory(sourceDir);
        }
    }

    private void processDirectory(File directory) {
        // 获取所有 Java 源文件
        List<File> javaFiles = new ArrayList<>();
        try (Stream<Path> paths = Files.walk(directory.toPath())) {
            javaFiles.addAll(paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .map(Path::toFile)
                    .collect(Collectors.toList()));
        } catch (IOException e) {
            logger.error("无法获取 Java 源文件", e);
        }

        // 计算相对路径
        Path relativePath = config.getProjectRoot().toPath().relativize(directory.toPath());
        outputDir = new File(config.getOutputDirectory(), relativePath.toString());

        // 创建输出目录
        File classInfoFile = new File(outputDir, "class_info.csv");
        File dependencyDir = new File(outputDir, "dependency");
        if (!dependencyDir.exists()) {
            dependencyDir.mkdirs();
        } else {
            // 跳过已处理的目录
            System.out.println("Skipping directory: " + relativePath);
            return;
        }

        // 打开 class_info CSV 文件写入器
        try {
            classInfoWriter = new BufferedWriter(new FileWriter(classInfoFile));
            classInfoWriter.write("className,filePath,startLine,endLine,simplifiedRepresentation\n");
            classInfoWriter.flush();
        } catch (IOException e) {
            logger.error("无法创建 class_info CSV 文件", e);
        }

        // 解析文件并应用Visitor
        System.out.println("Current Path: " + relativePath);
        try (ProgressBar pb = new ProgressBarBuilder()
                .setInitialMax(javaFiles.size())
                .setUpdateIntervalMillis(500)
                .continuousUpdate()
                .build();) {
            for (File file : javaFiles) {
                try {
                    CompilationUnit cu = javaParser.parse(file).getResult()
                            .orElseThrow(() -> new IOException("解析失败: " + file));
                    currentFilePath = file.getAbsolutePath();
                    visit(cu, new NodeContext(null, file.getName()));
                } catch (IOException e) {
                    logger.error("Error parsing file: " + file, e);
                }
                pb.step();
            }
        }

        // 关闭 class_info CSV 文件写入器
        try {
            if (classInfoWriter != null) {
                classInfoWriter.close();
            }
        } catch (IOException e) {
            logger.error("无法关闭 class_info CSV 文件", e);
        }
    }

    private void writeDependency(Dependency dependency) {
        if (dependencyWriter == null) {
            logger.error("Dependency writer is not initialized");
            return;
        }
        try {
            dependencyWriter.write(dependency.toCsvFormat() + "\n");
            dependencyWriter.flush();
        } catch (IOException e) {
            logger.error("无法写入依赖关系 CSV 文件", e);
        }
    }

    private void writeClassInfo(ClassInfo classInfo) {
        if (classInfoWriter == null) {
            logger.error("Class info writer is not initialized");
            return;
        }
        try {
            classInfoWriter.write(classInfo.toCsvFormat() + "\n");
            classInfoWriter.flush();
        } catch (IOException e) {
            logger.error("无法写入类信息 CSV 文件", e);
        }
    }

    @Override
    public void visit(ClassOrInterfaceDeclaration cid, NodeContext arg) {
        classStack.push(currentClassName);
        NodeContext parentContext = currentContext;
        try {
            currentClassName = cid.resolve().getQualifiedName();
        } catch (IllegalStateException e) {
            logger.warn("Symbol resolution not configured for class: {} in file {} at line {}", cid.getNameAsString(),
                    currentFilePath, getLineNumber(cid));
            classStack.pop();
            return;
        }
        currentContext = new NodeContext(parentContext, currentClassName);

        if (dependencyWriter != null) {
            dependencyWriterStack.push(dependencyWriter);
        }

        // 打开文件
        try {
            File dependencyFile = new File(outputDir, "dependency/" + currentClassName + ".csv");
            if (dependencyFile.exists()) {
                logger.warn("File already exists for class: " + currentClassName);
                return;
            }
            dependencyWriter = new BufferedWriter(new FileWriter(dependencyFile));
            dependencyWriter.write("source,target,type,filePath,lineNumber\n");
        } catch (IOException e) {
            logger.error("无法创建文件", e);
        }

        // 检测是否是内部类，并输出外层类与内部类的关系
        String outerClassName = classStack.peek();
        if (outerClassName != null) {
            writeDependency(
                    new InnerClassDependency(outerClassName, currentClassName, currentFilePath, getLineNumber(cid)));
        }

        int startLine = cid.getBegin().map(pos -> pos.line).orElse(-1);
        int endLine = cid.getEnd().map(pos -> pos.line).orElse(-1);
        String simplified = generateSimplifiedRepresentation(cid);
        writeClassInfo(new ClassInfo(currentClassName, currentFilePath, startLine, endLine, simplified));

        // 处理继承和接口实现
        super.visit(cid, arg);

        // 处理继承关系
        cid.getExtendedTypes().forEach(type -> {
            try {
                String parentClassName = resolveQualifiedTypeName(type);
                if (parentClassName != null) { // 过滤掉基本类型和 void
                    writeDependency(new InheritanceDependency(currentClassName, parentClassName, currentFilePath,
                            getLineNumber(type)));
                }
            } catch (UnsolvedSymbolException e) {
                logger.warn("无法解析继承关系: {} -> {} in file {} at line {}", currentClassName, type, currentFilePath,
                        getLineNumber(type));
            }
        });

        // 处理接口实现
        cid.getImplementedTypes().forEach(type -> {
            try {
                String interfaceClassName = resolveQualifiedTypeName(type);
                if (interfaceClassName != null) { // 过滤掉基本类型和 void
                    writeDependency(new InterfaceDependency(currentClassName, interfaceClassName, currentFilePath,
                            getLineNumber(type)));
                }
            } catch (UnsolvedSymbolException e) {
                logger.warn("无法解析接口实现: {} -> {} in file {} at line {}", currentClassName, type, currentFilePath,
                        getLineNumber(type));
            }
        });

        // 恢复外层类名
        currentClassName = classStack.pop();
        currentContext = parentContext;

        // 关闭文件
        try {
            if (dependencyWriter != null) {
                dependencyWriter.close();
            }
        } catch (IOException e) {
            logger.error("无法关闭文件", e);
        }

        // 恢复之前的 BufferedWriter
        if (!dependencyWriterStack.isEmpty()) {
            dependencyWriter = dependencyWriterStack.pop();
        } else {
            dependencyWriter = null;
        }
    }

    private String generateSimplifiedRepresentation(ClassOrInterfaceDeclaration cid) {
        StringBuilder sb = new StringBuilder();

        // 类声明
        String classModifiers = cid.getModifiers().stream()
                .map(Modifier::toString)
                .collect(Collectors.joining(""));
        if (!classModifiers.isEmpty()) {
            sb.append(classModifiers);
        }
        sb.append("class ")
                .append(cid.getNameAsString())
                .append("{");

        // 字段声明
        cid.getFields().forEach(field -> {
            String fieldModifiers = field.getModifiers().stream()
                    .map(Modifier::toString)
                    .collect(Collectors.joining(""));
            field.getVariables().forEach(var -> {
                if (!fieldModifiers.isEmpty()) {
                    sb.append(fieldModifiers);
                }
                sb.append(field.getElementType().asString())
                        .append(" ")
                        .append(var.getNameAsString())
                        .append(";");
            });
        });

        // 方法声明（仅保留必要空格）
        cid.getMethods().forEach(md -> {
            String methodModifiers = md.getModifiers().stream()
                    .map(Modifier::toString)
                    .collect(Collectors.joining(""));
            if (!methodModifiers.isEmpty()) {
                sb.append(methodModifiers);
            }
            sb.append(md.getType().asString())
                    .append(" ")
                    .append(md.getNameAsString())
                    .append("(");
            sb.append(md.getParameters().stream()
                    .map(p -> p.getType().asString() + " " + p.getNameAsString())
                    .collect(Collectors.joining(", ")));
            sb.append(");");
        });

        sb.append("}");
        return sb.toString();
    }

    @Override
    public void visit(FieldDeclaration fd, NodeContext arg) {
        super.visit(fd, arg);

        fd.getVariables().forEach(variable -> {
            try {
                String fieldClassName = resolveQualifiedTypeName(variable.getType());
                if (fieldClassName != null) { // 过滤掉基本类型和 void
                    writeDependency(new FieldDependency(currentClassName, fieldClassName, currentFilePath,
                            getLineNumber(variable)));
                }
            } catch (UnsolvedSymbolException e) {
                logger.warn("无法解析字段类型: {} -> {} in file {} at line {}", currentClassName, variable.getType(),
                        currentFilePath, getLineNumber(variable));
            }
        });
    }

    @Override
    public void visit(MethodDeclaration md, NodeContext arg) {
        if (currentContext == null) {
            logger.error("Current context is null");
            System.out.println("");
            return;
        }
        NodeContext parentContext = currentContext;
        String methodIdentifier = getMethodIdentifier(md);
        currentContext = new NodeContext(parentContext, parentContext.getName() + "$" + methodIdentifier);

        super.visit(md, arg);

        // 方法参数类型
        md.getParameters().forEach(param -> {
            try {
                String paramClassName = resolveQualifiedTypeName(param.getType());
                if (paramClassName != null) { // 过滤掉基本类型和 void
                    writeDependency(new MethodParameterDependency(currentClassName, paramClassName, currentFilePath,
                            getLineNumber(param)));
                }
            } catch (UnsolvedSymbolException e) {
                logger.warn("无法解析方法参数类型: {} -> {} in file {} at line {}", currentClassName, param.getType(),
                        currentFilePath, getLineNumber(param));
            }
        });

        // 返回类型
        try {
            String returnClassName = resolveQualifiedTypeName(md.getType());
            if (returnClassName != null) { // 过滤掉基本类型和 void
                writeDependency(new MethodReturnDependency(currentClassName, returnClassName, currentFilePath,
                        getLineNumber(md.getType())));
            }
        } catch (UnsolvedSymbolException e) {
            logger.warn("无法解析方法返回类型: {} -> {} in file {} at line {}", currentClassName, md.getType(), currentFilePath,
                    getLineNumber(md.getType()));
        }

        // 异常类型
        md.getThrownExceptions().forEach(ex -> {
            try {
                String exceptionClassName = resolveQualifiedTypeName(ex);
                if (exceptionClassName != null) { // 过滤掉基本类型和 void
                    writeDependency(new ExceptionDependency(currentClassName, exceptionClassName, currentFilePath,
                            getLineNumber(ex)));
                }
            } catch (UnsolvedSymbolException e) {
                logger.warn("无法解析异常类型: {} -> {} in file {} at line {}", currentClassName, ex, currentFilePath,
                        getLineNumber(ex));
            }
        });

        currentContext = parentContext;
    }

    @Override
    public void visit(VariableDeclarator vd, NodeContext arg) {
        super.visit(vd, arg);

        // 检测泛型容器与类型参数的依赖（如 List<String> -> List 依赖 String）
        if (vd.getType() instanceof ClassOrInterfaceType) {
            ClassOrInterfaceType type = (ClassOrInterfaceType) vd.getType();
            type.getTypeArguments().ifPresent(args -> {
                args.forEach(ta -> {
                    try {
                        // 获取泛型容器类（如 List）
                        String containerClassName = resolveQualifiedTypeName(type);
                        // 获取类型参数（如 String）
                        String genericClassName = resolveQualifiedTypeName(ta);
                        if (containerClassName != null && genericClassName != null) { // 过滤掉基本类型和 void
                            writeDependency(new GenericDependency(containerClassName, genericClassName, currentFilePath,
                                    getLineNumber(ta)));
                        }
                    } catch (UnsolvedSymbolException e) {
                        logger.warn("无法解析泛型参数类型: {} -> {} in file {} at line {}", type, ta, currentFilePath,
                                getLineNumber(ta));
                    }
                });
            });
        }
    }

    @Override
    public void visit(VariableDeclarationExpr vde, NodeContext arg) {
        super.visit(vde, arg);

        vde.getVariables().forEach(var -> {
            try {
                String varClassName = resolveQualifiedTypeName(var.getType());
                if (varClassName != null) { // 过滤掉基本类型和 void
                    writeDependency(new LocalVariableDependency(currentClassName, varClassName, currentFilePath,
                            getLineNumber(vde)));
                }
            } catch (UnsolvedSymbolException e) {
                logger.warn("无法解析局部变量类型: {} -> {} in file {} at line {}", currentClassName, var.getType(),
                        currentFilePath, getLineNumber(vde));
            }
        });
    }

    @Override
    public void visit(ObjectCreationExpr oce, NodeContext arg) {
        super.visit(oce, arg);

        try {
            String objectClassName = resolveQualifiedTypeName(oce.getType());
            if (objectClassName != null) { // 过滤掉基本类型和 void
                writeDependency(new ObjectInstantiationDependency(currentClassName, objectClassName, currentFilePath,
                        getLineNumber(oce)));
            }
        } catch (UnsolvedSymbolException e) {
            logger.warn("无法解析对象实例化类型: {} -> {} in file {} at line {}", currentClassName, oce.getType(), currentFilePath,
                    getLineNumber(oce));
        }
    }

    @Override
    public void visit(CastExpr ce, NodeContext arg) {
        super.visit(ce, arg);

        try {
            String targetClassName = resolveQualifiedTypeName(ce.getType());
            if (targetClassName != null) { // 过滤掉基本类型和 void
                writeDependency(
                        new CastDependency(currentClassName, targetClassName, currentFilePath, getLineNumber(ce)));
            }
        } catch (UnsolvedSymbolException e) {
            logger.warn("无法解析类型转换: {} -> {} in file {} at line {}", currentClassName, ce.getType(), currentFilePath,
                    getLineNumber(ce));
        }
    }

    @Override
    public void visit(TypeParameter tp, NodeContext arg) {
        currentContext.addTypeParameter(tp.getNameAsString());
        super.visit(tp, arg);

        try {
            // 生成泛型参数唯一标识
            String uniqueParamName = resolveQualifiedTypeName(tp);
            if (uniqueParamName != null) {
                tp.getTypeBound().forEach(bound -> {
                    try {
                        String boundClassName = resolveQualifiedTypeName(bound);
                        if (boundClassName != null) { // 过滤掉基本类型和 void
                            // 输出约束关系：泛型参数 -> 约束类型
                            writeDependency(
                                    new TypeParameterDependency(uniqueParamName, boundClassName, currentFilePath,
                                            getLineNumber(bound)));
                        }
                    } catch (UnsolvedSymbolException e) {
                        logger.warn("无法解析泛型参数约束: {} -> {} in file {} at line {}", uniqueParamName, bound,
                                currentFilePath,
                                getLineNumber(bound));
                    }
                });
            }
        } catch (UnsolvedSymbolException e) {
            logger.warn("无法解析泛型参数: {} in file {} at line {}", tp.getNameAsString(), currentFilePath, getLineNumber(tp));
        }
    }

    @Override
    public void visit(FieldAccessExpr fae, NodeContext arg) {
        super.visit(fae, arg);

        try {
            if (!fae.getScope().isThisExpr()) {
                ResolvedType resolvedType = fae.getScope().calculateResolvedType();
                if (resolvedType.isPrimitive()) {
                    logger.warn("无法解析基本类型的字段访问: {} in file {} at line {}", fae, currentFilePath, getLineNumber(fae));
                } else if (fae.resolve().asField().isStatic()) {
                    String staticMember = fae.resolve().asField().declaringType().getQualifiedName();
                    writeDependency(new StaticMemberDependency(currentClassName, staticMember, currentFilePath,
                            getLineNumber(fae)));
                }
            }
        } catch (UnsolvedSymbolException e) {
            logger.warn("无法解析静态成员: {} -> {} in file {} at line {}", currentClassName, fae.getName(),
                    currentFilePath, getLineNumber(fae));
        } catch (Exception e) {
            logger.warn("无法解析字段访问: {} in file {} at line {}", fae, currentFilePath, getLineNumber(fae));
        }
    }

    private int getLineNumber(Node node) {
        return node.getBegin().map(p -> p.line).orElse(-1);
    }

    // 生成唯一标识的工具方法
    private String getUniqueTypeName(String typeName) {
        NodeContext context = currentContext;
        while (context != null) {
            if (context.getTypeParameters().contains(typeName)) {
                return context.getName() + "#" + typeName;
            }
            context = context.getParentNode();
        }
        return typeName;
    }

    private String getMethodIdentifier(MethodDeclaration md) {
        StringBuilder identifier = new StringBuilder(md.getNameAsString());
        identifier.append("(");
        md.getParameters().forEach(param -> {
            identifier.append(param.getType().asString()).append(", ");
        });
        if (!md.getParameters().isEmpty()) {
            identifier.setLength(identifier.length() - 2); // 移除最后一个逗号和空格
        }
        identifier.append(")");
        return identifier.toString();
    }

    // 获取类型的全限定名
    private String resolveQualifiedTypeName(Type type) throws UnsolvedSymbolException {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<String> future = executor.submit(() -> {
            if (type.isPrimitiveType() || type.isVoidType()) {
                return null; // 基本类型和 void 返回 null
            }
            if (type.isArrayType()) {
                // 递归解析数组的元素类型
                return resolveQualifiedTypeName(type.asArrayType().getComponentType());
            }
            if (type.isTypeParameter()) {
                // 泛型参数类型，生成唯一标识
                return getUniqueTypeName(type.asTypeParameter().getName().asString());
            } else if (type.isClassOrInterfaceType()) {
                ResolvedType resolvedType = type.resolve();
                if (resolvedType.isTypeVariable()) {
                    return getUniqueTypeName(resolvedType.asTypeVariable().describe());
                }
                if (resolvedType.isReferenceType()) {
                    return resolvedType.asReferenceType().getQualifiedName();
                }
            }
            return type.resolve().describe();
        });

        try {
            return future.get(20, TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            future.cancel(true);
            logger.error("解析类型时超时: {} in file {} at line {}", type, currentFilePath, getLineNumber(type));
            System.out.println("");
            throw new UnsolvedSymbolException("解析类型超时: " + type.toString());
        } catch (InterruptedException | ExecutionException e) {
            logger.warn("解析类型时发生错误: {} in file {} at line {}", e.getClass().getName(), currentFilePath,
                    getLineNumber(type));
            // System.out.println("");
            throw new UnsolvedSymbolException(type.toString());
        } finally {
            executor.shutdown();
        }
    }
}
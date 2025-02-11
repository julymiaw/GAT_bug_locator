package com.example.dependencies;

public class ExceptionDependency extends Dependency {

    public ExceptionDependency(String source, String target, String filePath, int lineNumber) {
        super(source, target, "异常类型依赖", filePath, lineNumber);
    }
}

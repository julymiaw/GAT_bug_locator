package com.example.dependencies;

public class LocalVariableDependency extends Dependency {

    public LocalVariableDependency(String source, String target, String filePath, int lineNumber) {
        super(source, target, "局部变量依赖", filePath, lineNumber);
    }
}

package com.example.dependencies;

public class CastDependency extends Dependency {

    public CastDependency(String source, String target, String filePath, int lineNumber) {
        super(source, target, "类型转换依赖", filePath, lineNumber);
    }
}
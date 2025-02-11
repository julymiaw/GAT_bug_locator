package com.example.dependencies;

public class FieldDependency extends Dependency {

    public FieldDependency(String source, String target, String filePath, int lineNumber) {
        super(source, target, "字段类型依赖", filePath, lineNumber);
    }
}
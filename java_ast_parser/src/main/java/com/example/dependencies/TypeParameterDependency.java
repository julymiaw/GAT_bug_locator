package com.example.dependencies;

public class TypeParameterDependency extends Dependency {

    public TypeParameterDependency(String source, String target, String filePath, int lineNumber) {
        super(source, target, "泛型约束依赖", filePath, lineNumber);
    }
}
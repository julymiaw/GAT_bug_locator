package com.example.dependencies;

public class GenericDependency extends Dependency {

    public GenericDependency(String source, String target, String filePath, int lineNumber) {
        super(source, target, "泛型参数依赖", filePath, lineNumber);
    }
}

package com.example.dependencies;

public class InheritanceDependency extends Dependency {

    public InheritanceDependency(String source, String target, String filePath, int lineNumber) {
        super(source, target, "继承依赖", filePath, lineNumber);
    }
}

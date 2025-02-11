package com.example.dependencies;

public class InnerClassDependency extends Dependency {

    public InnerClassDependency(String source, String target, String filePath, int lineNumber) {
        super(source, target, "内部类依赖", filePath, lineNumber);
    }
}

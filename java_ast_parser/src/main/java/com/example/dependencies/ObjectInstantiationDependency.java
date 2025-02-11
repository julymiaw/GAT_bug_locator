package com.example.dependencies;

public class ObjectInstantiationDependency extends Dependency {

    public ObjectInstantiationDependency(String source, String target, String filePath, int lineNumber) {
        super(source, target, "对象实例化依赖", filePath, lineNumber);
    }
}

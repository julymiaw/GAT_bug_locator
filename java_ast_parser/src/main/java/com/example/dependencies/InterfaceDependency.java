package com.example.dependencies;

public class InterfaceDependency extends Dependency {

    public InterfaceDependency(String source, String target, String filePath, int lineNumber) {
        super(source, target, "接口依赖", filePath, lineNumber);
    }
}

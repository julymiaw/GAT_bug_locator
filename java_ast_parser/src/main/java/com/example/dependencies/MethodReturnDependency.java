package com.example.dependencies;

public class MethodReturnDependency extends Dependency {

    public MethodReturnDependency(String source, String target, String filePath, int lineNumber) {
        super(source, target, "方法返回依赖", filePath, lineNumber);
    }
}

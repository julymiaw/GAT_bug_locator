package com.example.dependencies;

public class MethodParameterDependency extends Dependency {

    public MethodParameterDependency(String source, String target, String filePath, int lineNumber) {
        super(source, target, "方法参数依赖", filePath, lineNumber);
    }
}

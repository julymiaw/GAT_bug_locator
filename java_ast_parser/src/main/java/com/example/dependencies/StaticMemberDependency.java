package com.example.dependencies;

public class StaticMemberDependency extends Dependency {
    public StaticMemberDependency(String source, String target, String filePath, int lineNumber) {
        super(source, target, "静态引用依赖", filePath, lineNumber);
    }
}

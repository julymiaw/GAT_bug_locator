package com.example.dependencies;

public abstract class Dependency {
    protected String source;
    protected String target;
    protected String type;
    protected String filePath;
    protected int lineNumber;

    public Dependency(String source, String target, String type, String filePath, int lineNumber) {
        this.source = source;
        this.target = target;
        this.type = type;
        this.filePath = filePath;
        this.lineNumber = lineNumber;
    }

    public void describe() {
        System.out.printf("类 %s 依赖类 %s，类型：%s，文件路径：%s，行号：%d%n", source, target, type, filePath, lineNumber);
    }

    public String toCsvFormat() {
        return String.format("\"%s\",\"%s\",%s,%s,%d", source, target, type, filePath, lineNumber);
    }
}
package com.example;

public class ClassInfo {
    private String className;
    private String filePath;
    private int startLine;
    private int endLine;
    private String simplifiedRepresentation;

    public ClassInfo(String className, String filePath, int startLine, int endLine, String simplifiedRepresentation) {
        this.className = className;
        this.filePath = filePath;
        this.startLine = startLine;
        this.endLine = endLine;
        this.simplifiedRepresentation = simplifiedRepresentation;
    }

    public String toCsvFormat() {
        return String.format("\"%s\",\"%s\",%d,%d,\"%s\"",
                className, filePath, startLine, endLine, simplifiedRepresentation);
    }
}

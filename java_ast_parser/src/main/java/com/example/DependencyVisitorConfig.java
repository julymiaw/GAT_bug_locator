package com.example;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class DependencyVisitorConfig {
    private boolean enableReflectionTypeSolver = false;
    private boolean enableJarTypeSolver = false;
    private List<File> jarFiles = new ArrayList<>();
    private List<File> sourceDirectories = new ArrayList<>();
    private File outputDirectory;
    private List<File> directoriesToParse = new ArrayList<>();
    private File projectRoot;

    public boolean isEnableReflectionTypeSolver() {
        return enableReflectionTypeSolver;
    }

    public void setEnableReflectionTypeSolver(boolean enableReflectionTypeSolver) {
        this.enableReflectionTypeSolver = enableReflectionTypeSolver;
    }

    public boolean isEnableJarTypeSolver() {
        return enableJarTypeSolver;
    }

    public void setEnableJarTypeSolver(boolean enableJarTypeSolver) {
        this.enableJarTypeSolver = enableJarTypeSolver;
    }

    public List<File> getJarFiles() {
        return jarFiles;
    }

    public void setJarFiles(List<File> jarFiles) {
        if (enableJarTypeSolver) {
            this.jarFiles = jarFiles;
        }
    }

    public List<File> getSourceDirectories() {
        return sourceDirectories;
    }

    public void setSourceDirectories(List<File> sourceDirectories) {
        this.sourceDirectories = sourceDirectories;
    }

    public File getOutputDirectory() {
        return outputDirectory;
    }

    public void setOutputDirectory(File outputDirectory) {
        this.outputDirectory = outputDirectory;
    }

    public List<File> getDirectoriesToParse() {
        return directoriesToParse;
    }

    public void setDirectoriesToParse(List<File> directoriesToParse) {
        this.directoriesToParse = directoriesToParse;
    }

    public File getProjectRoot() {
        return projectRoot;
    }

    public void setProjectRoot(File projectRoot) {
        this.projectRoot = projectRoot;
    }
}
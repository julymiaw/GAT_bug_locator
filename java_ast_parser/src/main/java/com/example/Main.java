package com.example;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        // 配置 DependencyVisitorConfig
        DependencyVisitorConfig config = new DependencyVisitorConfig();

        // 从 javaRoots.txt 文件中读取 sourceDirectories
        List<File> sourceDirectories = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(
                new FileReader("output/tomcat_dataset/javaRoots.txt"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sourceDirectories.add(new File(line));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        config.setSourceDirectories(sourceDirectories);
        config.setOutputDirectory(new File("output/tomcat_dataset"));
        config.setDirectoriesToParse(sourceDirectories);
        config.setProjectRoot(new File("source/tomcat_dataset/"));

        // 创建 DependencyVisitor 并处理
        DependencyVisitor visitor = new DependencyVisitor(config);
        visitor.process();
    }
}

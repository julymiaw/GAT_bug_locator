import com.github.javaparser.JavaParser;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.expr.FieldAccessExpr;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.NoSuchElementException;

import static org.junit.jupiter.api.Assertions.*;

public class ThisExprResolutionTest {

        @Test
        public void testThisExprTypeResolutionFailure() throws IOException {
                Path tempDir = Files.createTempDirectory("java_parser_test");
                Path sourcePath = tempDir.resolve("com/example/test/ThisExprTester.java");
                Files.createDirectories(sourcePath.getParent());

                String code = "package com.example.test;\n" +
                                "public class ThisExprTester {\n" +
                                "    Test test;\n" +
                                "    void getClasses() {\n" +
                                "        class innerClass {\n" +
                                "            public void test() {\n" +
                                "                ThisExprTester.this.test.test();\n" +
                                "            }\n" +
                                "        }\n" +
                                "    }\n" +
                                "}\n" +
                                "class Test { void test() {} }";
                Files.writeString(sourcePath, code);

                JavaSymbolSolver solver = new JavaSymbolSolver(
                                new JavaParserTypeSolver(tempDir.toFile()));
                JavaParser parser = new JavaParser(
                                new ParserConfiguration().setSymbolResolver(solver));

                CompilationUnit cu = parser.parse(sourcePath.toFile()).getResult().get();
                FieldAccessExpr fae = cu.findFirst(FieldAccessExpr.class).orElseThrow();

                assertTrue(fae.getScope().isThisExpr(), "Should recognize This expression");
                assertThrows(NoSuchElementException.class,
                                () -> fae.getScope().calculateResolvedType(),
                                "Current symbol resolution cannot handle ThisExprTester.this");
        }
}
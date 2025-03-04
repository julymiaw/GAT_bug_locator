import com.github.javaparser.JavaParser;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

public class StackOverflowTest {

    @Test
    public void testStackOverflowError() throws IOException {
        Path tempDir = Files.createTempDirectory("java_parser_test");
        Path sourcePath = tempDir.resolve("p/A.java");
        Files.createDirectories(sourcePath.getParent());

        String code = "package p;\n" +
                "\n" +
                "class A {\n" +
                "    A a;\n" +
                "    class Inner {\n" +
                "    }\n" +
                "    void f(A a) {\n" +
                "        a.a.a.new Inner();\n" +
                "    }\n" +
                "}\n";
        Files.writeString(sourcePath, code);

        JavaSymbolSolver solver = new JavaSymbolSolver(
                new JavaParserTypeSolver(tempDir.toFile()));
        JavaParser parser = new JavaParser(
                new ParserConfiguration().setSymbolResolver(solver));

        CompilationUnit cu = parser.parse(sourcePath.toFile()).getResult().get();
        ObjectCreationExpr oce = cu.findFirst(ObjectCreationExpr.class).orElseThrow();

        assertThrows(StackOverflowError.class,
                () -> oce.resolve(),
                "Expected a StackOverflowError due to recursive resolution");
    }
}
package com.example;

import java.util.ArrayList;
import java.util.List;

public class NodeContext {
    private NodeContext parentNode;
    private String name;
    private List<String> typeParameters;

    public NodeContext(NodeContext parentNode, String name) {
        this.parentNode = parentNode;
        this.name = name;
        this.typeParameters = new ArrayList<>();
    }

    public NodeContext getParentNode() {
        return parentNode;
    }

    public String getName() {
        return name;
    }

    public List<String> getTypeParameters() {
        return typeParameters;
    }

    public void addTypeParameter(String typeParameter) {
        this.typeParameters.add(typeParameter);
    }
}
from graphviz import Digraph

def create_custom_flowchart():
    dot = Digraph(format='png')
    dot.attr(rankdir='LR')  # Left to Right layout

    # Input node (standalone)
    dot.node('Input', 'Input Layer\n(input_shape)', shape='box')

    # === Block 1 ===
    dot.node('Conv1_1', 'Conv2D (64)\n3x3 + ReLU + BN', shape='box')
    dot.node('Conv1_2', 'Conv2D (64)\n3x3 + ReLU + BN', shape='box')
    dot.node('Pool1', 'MaxPool 2x2\nDropout(0.25)', shape='box')

    # === Block 2 ===
    dot.node('Conv2_1', 'Conv2D (128)\n3x3 + ReLU + BN', shape='box')
    dot.node('Conv2_2', 'Conv2D (128)\n3x3 + ReLU + BN', shape='box')
    dot.node('Pool2', 'MaxPool 2x2\nDropout(0.25)', shape='box')

    # === Block 3 ===
    dot.node('Conv3_1', 'Conv2D (256)\n3x3 + ReLU + BN', shape='box')
    dot.node('Conv3_2', 'Conv2D (256)\n3x3 + ReLU + BN', shape='box')
    dot.node('Pool3', 'MaxPool 2x2\nDropout(0.25)', shape='box')

    # === Classifier ===
    dot.node('GAP', 'GlobalAvgPooling2D', shape='box')
    dot.node('FC1', 'Dense(256)\nReLU + BN\nDropout(0.5)', shape='box')
    dot.node('Output', 'Dense(43)\nSoftmax', shape='box')

    # === Layer connections ===
    dot.edge('Input', 'Conv1_1')
    dot.edge('Conv1_1', 'Conv1_2')
    dot.edge('Conv1_2', 'Pool1')
    dot.edge('Pool1', 'Conv2_1')
    dot.edge('Conv2_1', 'Conv2_2')
    dot.edge('Conv2_2', 'Pool2')

    # vertical connection to next row
    dot.edge('Pool2', 'Conv3_1', constraint='false')  # vertical jump
    dot.edge('Conv3_1', 'Conv3_2')
    dot.edge('Conv3_2', 'Pool3')
    dot.edge('Pool3', 'GAP')
    dot.edge('GAP', 'FC1')
    dot.edge('FC1', 'Output')

    # === Rank alignments ===
    dot.body.append('{ rank = same; Conv1_1; Conv1_2; Pool1; Conv2_1; Conv2_2; Pool2; }')
    dot.body.append('{ rank = same; Conv3_1; Conv3_2; Pool3; GAP; FC1; Output; }')

    return dot

# Render the graph
flowchart = create_custom_flowchart()
flowchart.render('traffic_sign_net_custom_flowchart', view=True)

//
//  neuralnetTests.m
//  neuralnetTests
//
//  Created by Adam Wulf on 10/9/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import <XCTest/XCTest.h>
#import <neuralnet/neuralnet.h>
#import "AbstractNeuron+Protected.h"

@interface neuralnetTests : XCTestCase

@end

@implementation neuralnetTests

- (void)setUp
{
    [super setUp];
    // Put setup code here. This method is called before the invocation of each test method in the class.
}

- (void)tearDown
{
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    [super tearDown];
}

// test case from Grokking Deep Learning book by Andrew W. Trask
- (void)testChapter3Page24
{
    StaticNeuron *input = [[StaticNeuron alloc] initWithValue:8.5];
    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];
    [output addInput:input withWeight:0.1];

    // run the neural net
    [output forwardPass];

    XCTAssertEqualWithAccuracy(.85, [output activation], .0000000000001, @"prediction is correct");
}

// test case from Grokking Deep Learning book by Andrew W. Trask
- (void)testChapter3Page34
{
    StaticNeuron *i1 = [[StaticNeuron alloc] initWithValue:8.5];
    StaticNeuron *i2 = [[StaticNeuron alloc] initWithValue:.65];
    StaticNeuron *i3 = [[StaticNeuron alloc] initWithValue:1.2];
    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];
    [output addInput:i1 withWeight:0.1];
    [output addInput:i2 withWeight:0.2];
    [output addInput:i3 withWeight:0];

    // run the neural net
    [output forwardPass];

    XCTAssertEqualWithAccuracy(.98, [output activation], .0000000000001, @"prediction is correct");
}

// test case from Grokking Deep Learning book by Andrew W. Trask
- (void)testChapter3Page37
{
    StaticNeuron *i1 = [[StaticNeuron alloc] initWithValue:.65];
    WeightedSumNeuron *output1 = [[WeightedSumNeuron alloc] init];
    WeightedSumNeuron *output2 = [[WeightedSumNeuron alloc] init];
    WeightedSumNeuron *output3 = [[WeightedSumNeuron alloc] init];

    [output1 addInput:i1 withWeight:0.3];
    [output2 addInput:i1 withWeight:0.2];
    [output3 addInput:i1 withWeight:0.9];

    // run the neural net
    [output1 forwardPass];
    [output2 forwardPass];
    [output3 forwardPass];

    XCTAssertEqualWithAccuracy(.195, [output1 activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(.13, [output2 activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(.585, [output3 activation], .0000000000001, @"prediction is correct");
}

// test case from Grokking Deep Learning book by Andrew W. Trask
- (void)testChapter3Page39
{
    StaticNeuron *i1 = [[StaticNeuron alloc] initWithValue:8.5];
    StaticNeuron *i2 = [[StaticNeuron alloc] initWithValue:.65];
    StaticNeuron *i3 = [[StaticNeuron alloc] initWithValue:1.2];
    WeightedSumNeuron *output1 = [[WeightedSumNeuron alloc] init];
    WeightedSumNeuron *output2 = [[WeightedSumNeuron alloc] init];
    WeightedSumNeuron *output3 = [[WeightedSumNeuron alloc] init];

    [output1 addInput:i1 withWeight:0.1];
    [output1 addInput:i2 withWeight:0.1];
    [output1 addInput:i3 withWeight:-0.3];

    [output2 addInput:i1 withWeight:0.1];
    [output2 addInput:i2 withWeight:0.2];
    [output2 addInput:i3 withWeight:0.0];

    [output3 addInput:i1 withWeight:0.0];
    [output3 addInput:i2 withWeight:1.3];
    [output3 addInput:i3 withWeight:0.1];

    // run the neural net
    [output1 forwardPass];
    [output2 forwardPass];
    [output3 forwardPass];

    XCTAssertEqualWithAccuracy(.555, [output1 activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(.98, [output2 activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(.965, [output3 activation], .0000000000001, @"prediction is correct");
}

// test case from Grokking Deep Learning book by Andrew W. Trask
- (void)testChapter3Page43
{
    StaticNeuron *i1 = [[StaticNeuron alloc] initWithValue:8.5];
    StaticNeuron *i2 = [[StaticNeuron alloc] initWithValue:.65];
    StaticNeuron *i3 = [[StaticNeuron alloc] initWithValue:1.2];

    WeightedSumNeuron *hidden1 = [[WeightedSumNeuron alloc] init];
    WeightedSumNeuron *hidden2 = [[WeightedSumNeuron alloc] init];
    WeightedSumNeuron *hidden3 = [[WeightedSumNeuron alloc] init];

    WeightedSumNeuron *output1 = [[WeightedSumNeuron alloc] init];
    WeightedSumNeuron *output2 = [[WeightedSumNeuron alloc] init];
    WeightedSumNeuron *output3 = [[WeightedSumNeuron alloc] init];

    [hidden1 addInput:i1 withWeight:0.1];
    [hidden1 addInput:i2 withWeight:0.2];
    [hidden1 addInput:i3 withWeight:-0.1];

    [hidden2 addInput:i1 withWeight:-0.1];
    [hidden2 addInput:i2 withWeight:0.1];
    [hidden2 addInput:i3 withWeight:0.9];

    [hidden3 addInput:i1 withWeight:0.1];
    [hidden3 addInput:i2 withWeight:0.4];
    [hidden3 addInput:i3 withWeight:0.1];

    [output1 addInput:hidden1 withWeight:0.3];
    [output1 addInput:hidden2 withWeight:1.1];
    [output1 addInput:hidden3 withWeight:-0.3];

    [output2 addInput:hidden1 withWeight:0.1];
    [output2 addInput:hidden2 withWeight:0.2];
    [output2 addInput:hidden3 withWeight:0.0];

    [output3 addInput:hidden1 withWeight:0.0];
    [output3 addInput:hidden2 withWeight:1.3];
    [output3 addInput:hidden3 withWeight:0.1];

    // run the neural net
    [hidden1 forwardPass];
    [hidden2 forwardPass];
    [hidden3 forwardPass];

    [output1 forwardPass];
    [output2 forwardPass];
    [output3 forwardPass];

    XCTAssertEqualWithAccuracy(.86, [hidden1 activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(.295, [hidden2 activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(1.23, [hidden3 activation], .0000000000001, @"prediction is correct");

    // note, the book rounds outputs 1 and 3 to 3 digits
    XCTAssertEqualWithAccuracy(.2135, [output1 activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(.145, [output2 activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(.5065, [output3 activation], .0000000000001, @"prediction is correct");
}

// test case from Grokking Deep Learning book by Andrew W. Trask
- (void)testChapter4Page50
{
    StaticNeuron *input = [[StaticNeuron alloc] initWithValue:0.5];
    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];
    [output addInput:input withWeight:0.5];

    // run the neural net
    [output forwardPass];

    const CGFloat kGoal = 0.8;

    XCTAssertEqualWithAccuracy(0.25, [output activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(-0.55, [output simpleErrorFor:kGoal], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.3025, [output errorFor:kGoal], .0000000000001, @"prediction is correct");
}

// test case from Grokking Deep Learning book by Andrew W. Trask
- (void)testChapter4Page57
{
    StaticNeuron *input = [[StaticNeuron alloc] initWithValue:.5];
    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];
    [output addInput:input withWeight:0.5];

    const CGFloat kGoal = 0.8;

    // run the neural net
    [output forwardPass];

    XCTAssertEqualWithAccuracy(0.25, [output activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(-0.55, [output simpleErrorFor:kGoal], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.3025, [output errorFor:kGoal], .0000000000001, @"prediction is correct");

    [output backpropagateFor:kGoal];
    [output updateWeightsWithAlpha:1.0];
    [output forwardPass];

    XCTAssertEqualWithAccuracy(0.3875, [output activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.17015625, [output errorFor:kGoal], .0000000000001, @"prediction is correct");

    [output backpropagateFor:kGoal];
    [output updateWeightsWithAlpha:1.0];
    [output forwardPass];

    XCTAssertEqualWithAccuracy(0.490625, [output activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.095712890625, [output errorFor:kGoal], .0000000000001, @"prediction is correct");

    for (int i = 4; i <= 20; i++) {
        [output backpropagateFor:kGoal];
        [output updateWeightsWithAlpha:1.0];
        [output forwardPass];
    }

    // the book truncates the output number
    XCTAssertEqualWithAccuracy(0.79767444457811509, [output activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.00000540820802026, [output errorFor:kGoal], .0000000000001, @"prediction is correct");
}

// test case from Grokking Deep Learning book by Andrew W. Trask
- (void)testChapter4Page59
{
    StaticNeuron *input = [[StaticNeuron alloc] initWithValue:8.5];
    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];
    [output addInput:input withWeight:0.1];

    // run the neural net
    [output forwardPass];

    const CGFloat kGoal = 1.0;

    XCTAssertEqualWithAccuracy(0.85, [output activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(-0.15, [output simpleErrorFor:kGoal], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.0225, [output errorFor:kGoal], .0000000000001, @"prediction is correct");

    [output backpropagateFor:kGoal];
    [output updateWeightsWithAlpha:0.01];

    XCTAssertEqualWithAccuracy(0.11275, [output weightForNeuron:input], .0000000000001, @"prediction is correct");
}

// test case from Grokking Deep Learning book by Andrew W. Trask
- (void)testChapter4Page62
{
    StaticNeuron *input = [[StaticNeuron alloc] initWithValue:1.1];
    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];
    [output addInput:input withWeight:0.0];

    // run the neural net
    [output forwardPass];

    const CGFloat kGoal = 0.8;

    XCTAssertEqualWithAccuracy(0.0, [output activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(-0.8, [output simpleErrorFor:kGoal], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.64, [output errorFor:kGoal], .0000000000001, @"prediction is correct");

    [output backpropagateFor:kGoal];
    [output updateWeightsWithAlpha:1.0];

    XCTAssertEqualWithAccuracy(0.88, [output weightForNeuron:input], .0000000000001, @"prediction is correct");

    [output forwardPass];
    [output backpropagateFor:kGoal];
    [output updateWeightsWithAlpha:1.0];

    // the book rounds this value
    XCTAssertEqualWithAccuracy(0.6952, [output weightForNeuron:input], .0000000000001, @"prediction is correct");

    [output forwardPass];
    [output backpropagateFor:kGoal];
    [output updateWeightsWithAlpha:1.0];

    // the book rounds this value
    XCTAssertEqualWithAccuracy(0.734008, [output weightForNeuron:input], .0000000000001, @"prediction is correct");

    [output forwardPass];
    [output backpropagateFor:kGoal];
    [output updateWeightsWithAlpha:1.0];

    // the book rounds this value
    XCTAssertEqualWithAccuracy(0.72585832, [output weightForNeuron:input], .0000000000001, @"prediction is correct");
}

// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
- (void)testMattMazurExample
{
    StaticNeuron *input1 = [[StaticNeuron alloc] initWithValue:0.05];
    StaticNeuron *input2 = [[StaticNeuron alloc] initWithValue:0.1];
    StaticNeuron *bias = [[StaticNeuron alloc] initWithValue:1.0];

    SigmoidNeuron *hidden1 = [[SigmoidNeuron alloc] init];
    [hidden1 addInput:input1 withWeight:0.15];
    [hidden1 addInput:input2 withWeight:0.2];
    [hidden1 addInput:bias withWeight:0.35];

    SigmoidNeuron *hidden2 = [[SigmoidNeuron alloc] init];
    [hidden2 addInput:input1 withWeight:0.25];
    [hidden2 addInput:input2 withWeight:0.3];
    [hidden2 addInput:bias withWeight:0.35];

    SigmoidNeuron *output1 = [[SigmoidNeuron alloc] init];
    [output1 addInput:hidden1 withWeight:0.4];
    [output1 addInput:hidden2 withWeight:0.45];
    [output1 addInput:bias withWeight:0.6];

    SigmoidNeuron *output2 = [[SigmoidNeuron alloc] init];
    [output2 addInput:hidden1 withWeight:0.5];
    [output2 addInput:hidden2 withWeight:0.55];
    [output2 addInput:bias withWeight:0.6];

    const CGFloat alpha = 0.5;

    // run the neural net
    void (^forwardPass)(void) = ^{
        [hidden1 forwardPass];
        [hidden2 forwardPass];
        [output1 forwardPass];
        [output2 forwardPass];
    };

    void (^backwardPass)(CGFloat, CGFloat) = ^(CGFloat goal1, CGFloat goal2) {
        [output1 backpropagateFor:goal1];
        [output2 backpropagateFor:goal2];
        [hidden1 backpropagate];
        [hidden2 backpropagate];
        [output1 updateWeightsWithAlpha:alpha];
        [output2 updateWeightsWithAlpha:alpha];
        [hidden1 updateWeightsWithAlpha:alpha];
        [hidden2 updateWeightsWithAlpha:alpha];
    };

    forwardPass();
    backwardPass(0.01, 0.99);

    XCTAssertEqualWithAccuracy(0.1497807161327628, [hidden1 weightForNeuron:input1], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.1995614322655257, [hidden1 weightForNeuron:input2], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.2497511436323696, [hidden2 weightForNeuron:input1], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.2995022872647392, [hidden2 weightForNeuron:input2], .0000000000001, @"prediction is correct");

    XCTAssertEqualWithAccuracy(0.3589164797178847, [output1 weightForNeuron:hidden1], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.4086661860762334, [output1 weightForNeuron:hidden2], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.5113012702387375, [output2 weightForNeuron:hidden1], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.5613701211079891, [output2 weightForNeuron:hidden2], .0000000000001, @"prediction is correct");
}

@end

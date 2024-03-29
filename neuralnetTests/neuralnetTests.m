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
    srand(1);
    // Put setup code here. This method is called before the invocation of each test method in the class.
}

- (void)tearDown
{
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    [super tearDown];
}

- (void)testRandomGenerator {
    NSMutableArray *generated = [NSMutableArray array];
    srand(1);

    for (NSInteger i=0; i<10000; i++) {
        int num = rand();
        [generated addObject:@(num)];

        NSLog(@"%d", num);
    }

    NSData* data = [NSJSONSerialization dataWithJSONObject:generated options:NSJSONWritingPrettyPrinted error:nil];
    NSArray *arr = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];

    XCTAssertEqualObjects(arr, generated);
    XCTAssertEqualObjects(arr[100], generated[100]);
    XCTAssertEqualObjects(arr[200], generated[200]);
    XCTAssertEqualObjects(arr[300], generated[300]);
    XCTAssertEqualObjects(arr[1000], generated[1000]);
    XCTAssertEqualObjects(arr[2000], generated[2000]);
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
    ErrorCalculator *error = [[TraskMeanSquaredError alloc] init];
    StaticNeuron *input = [[StaticNeuron alloc] initWithValue:0.5];
    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];
    [output addInput:input withWeight:0.5];

    // run the neural net
    [output forwardPass];

    const CGFloat kGoal = 0.8;

    XCTAssertEqualWithAccuracy(0.25, [output activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.55, [error simpleErrorFor:kGoal forNeuron:output], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.3025, [error errorFor:kGoal forNeuron:output], .0000000000001, @"prediction is correct");
}

// test case from Grokking Deep Learning book by Andrew W. Trask
- (void)testChapter4Page57
{
    ErrorCalculator *error = [[TraskMeanSquaredError alloc] init];
    StaticNeuron *input = [[StaticNeuron alloc] initWithValue:.5];
    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];
    [output addInput:input withWeight:0.5];

    const CGFloat kGoal = 0.8;

    // run the neural net
    [output forwardPass];

    XCTAssertEqualWithAccuracy(0.25, [output activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.55, [error simpleErrorFor:kGoal forNeuron:output], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.3025, [error errorFor:kGoal forNeuron:output], .0000000000001, @"prediction is correct");

    [output backpropagateFor:[error errorDerivativeFor:kGoal forNeuron:output]];
    [output updateWeightsWithAlpha:1.0];
    [output forwardPass];

    XCTAssertEqualWithAccuracy(0.3875, [output activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.17015625, [error errorFor:kGoal forNeuron:output], .0000000000001, @"prediction is correct");

    [output backpropagateFor:[error errorDerivativeFor:kGoal forNeuron:output]];
    [output updateWeightsWithAlpha:1.0];
    [output forwardPass];

    XCTAssertEqualWithAccuracy(0.490625, [output activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.095712890625, [error errorFor:kGoal forNeuron:output], .0000000000001, @"prediction is correct");

    for (int i = 4; i <= 20; i++) {
        [output backpropagateFor:[error errorDerivativeFor:kGoal forNeuron:output]];
        [output updateWeightsWithAlpha:1.0];
        [output forwardPass];
    }

    // the book truncates the output number
    XCTAssertEqualWithAccuracy(0.79767444457811509, [output activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.00000540820802026, [error errorFor:kGoal forNeuron:output], .0000000000001, @"prediction is correct");
}

// test case from Grokking Deep Learning book by Andrew W. Trask
- (void)testChapter4Page59
{
    ErrorCalculator *error = [[TraskMeanSquaredError alloc] init];
    StaticNeuron *input = [[StaticNeuron alloc] initWithValue:8.5];
    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];
    [output addInput:input withWeight:0.1];

    // run the neural net
    [output forwardPass];

    const CGFloat kGoal = 1.0;

    XCTAssertEqualWithAccuracy(0.85, [output activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.15, [error simpleErrorFor:kGoal forNeuron:output], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.0225, [error errorFor:kGoal forNeuron:output], .0000000000001, @"prediction is correct");

    [output backpropagateFor:[error errorDerivativeFor:kGoal forNeuron:output]];
    [output updateWeightsWithAlpha:0.01];

    XCTAssertEqualWithAccuracy(0.11275, [output weightForNeuron:input], .0000000000001, @"prediction is correct");
}

// test case from Grokking Deep Learning book by Andrew W. Trask
- (void)testChapter4Page62
{
    ErrorCalculator *error = [[TraskMeanSquaredError alloc] init];
    StaticNeuron *input = [[StaticNeuron alloc] initWithValue:1.1];
    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];
    [output addInput:input withWeight:0.0];

    // run the neural net
    [output forwardPass];

    const CGFloat kGoal = 0.8;

    XCTAssertEqualWithAccuracy(0.0, [output activation], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.8, [error simpleErrorFor:kGoal forNeuron:output], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(0.64, [error errorFor:kGoal forNeuron:output], .0000000000001, @"prediction is correct");

    [output backpropagateFor:[error errorDerivativeFor:kGoal forNeuron:output]];
    [output updateWeightsWithAlpha:1.0];

    XCTAssertEqualWithAccuracy(0.88, [output weightForNeuron:input], .0000000000001, @"prediction is correct");

    [output forwardPass];
    [output backpropagateFor:[error errorDerivativeFor:kGoal forNeuron:output]];
    [output updateWeightsWithAlpha:1.0];

    // the book rounds this value
    XCTAssertEqualWithAccuracy(0.6952, [output weightForNeuron:input], .0000000000001, @"prediction is correct");

    [output forwardPass];
    [output backpropagateFor:[error errorDerivativeFor:kGoal forNeuron:output]];
    [output updateWeightsWithAlpha:1.0];

    // the book rounds this value
    XCTAssertEqualWithAccuracy(0.734008, [output weightForNeuron:input], .0000000000001, @"prediction is correct");

    [output forwardPass];
    [output backpropagateFor:[error errorDerivativeFor:kGoal forNeuron:output]];
    [output updateWeightsWithAlpha:1.0];

    // the book rounds this value
    XCTAssertEqualWithAccuracy(0.72585832, [output weightForNeuron:input], .0000000000001, @"prediction is correct");
}

// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
- (void)testMattMazurExample
{
    ErrorCalculator *error = [[TraskMeanSquaredError alloc] init];
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
        [output1 backpropagateFor:[error errorDerivativeFor:goal1 forNeuron:output1]];
        [output2 backpropagateFor:[error errorDerivativeFor:goal2 forNeuron:output2]];
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

// test for xor - this should fail since these are all linear neurons
- (void)testLinearXOR
{
    CGFloat (^randF)(void) = ^{
        return (rand() % 20000 - 10000.0) / 10000.0;
    };

    ErrorCalculator *error = [[TraskMeanSquaredError alloc] init];
    InputNeuron *leftInput = [[InputNeuron alloc] initWithValue:1.0];
    InputNeuron *rightInput = [[InputNeuron alloc] initWithValue:0.0];
    StaticNeuron *bias = [[StaticNeuron alloc] initWithValue:1.0];

    WeightedSumNeuron *hidden1 = [[WeightedSumNeuron alloc] init];
    [hidden1 addInput:leftInput withWeight:randF()];
    [hidden1 addInput:rightInput withWeight:randF()];

    WeightedSumNeuron *hidden2 = [[WeightedSumNeuron alloc] init];
    [hidden2 addInput:leftInput withWeight:randF()];
    [hidden2 addInput:rightInput withWeight:randF()];

    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];
    [output addInput:hidden1 withWeight:randF()];
    [output addInput:hidden2 withWeight:randF()];

    [hidden1 addInput:bias withWeight:randF()];
    [hidden2 addInput:bias withWeight:randF()];
    [output addInput:bias withWeight:randF()];

    const CGFloat alpha = 0.025;

    // run the neural net
    void (^forwardPass)(void) = ^{
        [hidden1 forwardPass];
        [hidden2 forwardPass];
        [output forwardPass];
    };

    void (^backwardPass)(CGFloat) = ^(CGFloat goal) {
        [output backpropagateFor:[error errorDerivativeFor:goal forNeuron:output]];
        [hidden1 backpropagate];
        [hidden2 backpropagate];
        [output updateWeightsWithAlpha:alpha];
        [hidden1 updateWeightsWithAlpha:alpha];
        [hidden2 updateWeightsWithAlpha:alpha];
    };

    NSArray *data = @[@[@0, @0, @0],
                      @[@1, @0, @1],
                      @[@0, @1, @1],
                      @[@1, @1, @0]];

    CGFloat avgError = 0;

    for (NSInteger i = 0; i < 10000; i++) {
        NSArray *testCase = data[i % [data count]];

        [leftInput setActivation:[testCase[0] doubleValue]];
        [rightInput setActivation:[testCase[1] doubleValue]];

        forwardPass();
        backwardPass([testCase[2] doubleValue]);

        avgError = avgError * 0.9 + ABS([error errorFor:[testCase[2] doubleValue] forNeuron:output]) * 0.1;
    }

    XCTAssertEqualWithAccuracy(avgError, .25, .01);
}

// test for xor - this should fail since these are all linear neurons
- (void)testSigmoidXOR
{
    CGFloat (^randF)(void) = ^{
        return (rand() % 20000 - 10000.0) / 10000.0;
    };

    ErrorCalculator *error = [[TraskMeanSquaredError alloc] init];
    InputNeuron *leftInput = [[InputNeuron alloc] initWithValue:1.0];
    InputNeuron *rightInput = [[InputNeuron alloc] initWithValue:0.0];
    StaticNeuron *bias = [[StaticNeuron alloc] initWithValue:1.0];

    SigmoidNeuron *hidden1 = [[SigmoidNeuron alloc] init];
    [hidden1 addInput:leftInput withWeight:randF()];
    [hidden1 addInput:rightInput withWeight:randF()];

    SigmoidNeuron *hidden2 = [[SigmoidNeuron alloc] init];
    [hidden2 addInput:leftInput withWeight:randF()];
    [hidden2 addInput:rightInput withWeight:randF()];

    SigmoidNeuron *output = [[SigmoidNeuron alloc] init];
    [output addInput:hidden1 withWeight:randF()];
    [output addInput:hidden2 withWeight:randF()];

    [hidden1 addInput:bias withWeight:randF()];
    [hidden2 addInput:bias withWeight:randF()];
    [output addInput:bias withWeight:randF()];

    const CGFloat alpha = 0.1;

    // run the neural net
    void (^forwardPass)(void) = ^{
        [hidden1 forwardPass];
        [hidden2 forwardPass];
        [output forwardPass];
    };

    void (^backwardPass)(CGFloat) = ^(CGFloat goal) {
        [output backpropagateFor:[error errorDerivativeFor:goal forNeuron:output]];
        [hidden1 backpropagate];
        [hidden2 backpropagate];
        [output updateWeightsWithAlpha:alpha];
        [hidden1 updateWeightsWithAlpha:alpha];
        [hidden2 updateWeightsWithAlpha:alpha];
    };

    NSArray *data = @[@[@0, @0, @0],
                      @[@1, @0, @1],
                      @[@0, @1, @1],
                      @[@1, @1, @0]];

    CGFloat avgError = 0;

    for (NSInteger i = 0; i < 100000; i++) {
        NSArray *testCase = data[i % [data count]];

        [leftInput setActivation:[testCase[0] doubleValue]];
        [rightInput setActivation:[testCase[1] doubleValue]];

        forwardPass();
        backwardPass([testCase[2] doubleValue]);

        avgError = avgError * 0.9 + ABS([error errorFor:[testCase[2] doubleValue] forNeuron:output]) * 0.1;
    }

    XCTAssertEqualWithAccuracy(avgError, 0, .01);

    for (NSInteger i = 0; i < [data count]; i++) {
        NSArray *testCase = data[i % [data count]];

        [leftInput setActivation:[testCase[0] doubleValue]];
        [rightInput setActivation:[testCase[1] doubleValue]];

        forwardPass();

        XCTAssertEqualWithAccuracy([output activation], [testCase[2] doubleValue], .1);
    }
}

// test for xor - this should fail since these are all linear neurons
- (void)testSigmoidXORWithBadRandomWeights
{
    ErrorCalculator *error = [[TraskMeanSquaredError alloc] init];
    InputNeuron *leftInput = [[InputNeuron alloc] initWithValue:1.0];
    InputNeuron *rightInput = [[InputNeuron alloc] initWithValue:0.0];
    StaticNeuron *bias = [[StaticNeuron alloc] initWithValue:1.0];

    SigmoidNeuron *hidden1 = [[SigmoidNeuron alloc] init];
    [hidden1 addInput:leftInput withWeight:0.1651];
    [hidden1 addInput:rightInput withWeight:-0.6032999999999999];
    [hidden1 addInput:bias withWeight:-0.4913];

    SigmoidNeuron *hidden2 = [[SigmoidNeuron alloc] init];
    [hidden2 addInput:leftInput withWeight:0.3267];
    [hidden2 addInput:rightInput withWeight:-0.0517];
    [hidden2 addInput:bias withWeight:-0.5953000000000001];

    // interestingly, if this last output neuron is a sigmoid,
    // then the test fails as the network will get stuck
    // in a bad position.
    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];
    [output addInput:hidden1 withWeight:-0.3543];
    [output addInput:hidden2 withWeight:0.3965];
    [output addInput:bias withWeight:-0.3629];

    const CGFloat alpha = 0.1;

    // run the neural net
    void (^forwardPass)(void) = ^{
        [hidden1 forwardPass];
        [hidden2 forwardPass];
        [output forwardPass];
    };

    void (^backwardPass)(CGFloat) = ^(CGFloat goal) {
        [output backpropagateFor:[error errorDerivativeFor:goal forNeuron:output]];
        [hidden1 backpropagate];
        [hidden2 backpropagate];
        [output updateWeightsWithAlpha:alpha];
        [hidden1 updateWeightsWithAlpha:alpha];
        [hidden2 updateWeightsWithAlpha:alpha];
    };

    NSArray *data = @[@[@0, @0, @0],
                      @[@1, @0, @1],
                      @[@0, @1, @1],
                      @[@1, @1, @0]];

    CGFloat avgError = 0;

    for (NSInteger i = 0; i < 100000; i++) {
        NSArray *testCase = data[i % [data count]];

        [leftInput setActivation:[testCase[0] doubleValue]];
        [rightInput setActivation:[testCase[1] doubleValue]];

        forwardPass();
        backwardPass([testCase[2] doubleValue]);

        avgError = avgError * 0.9 + ABS([error errorFor:[testCase[2] doubleValue] forNeuron:output]) * 0.1;
    }

    XCTAssertEqualWithAccuracy(avgError, 0, .01);

    for (NSInteger i = 0; i < [data count]; i++) {
        NSArray *testCase = data[i % [data count]];

        [leftInput setActivation:[testCase[0] doubleValue]];
        [rightInput setActivation:[testCase[1] doubleValue]];

        forwardPass();

        XCTAssertEqualWithAccuracy([output activation], [testCase[2] doubleValue], .1);
    }
}

// test case from Grokking Deep Learning book by Andrew W. Trask
- (void)testChapter6Page128
{
    CGFloat (^randF)(void) = ^{
        return (rand() % 20000 - 10000.0) / 10000.0;
    };

    ErrorCalculator *error = [[TraskMeanSquaredError alloc] init];
    InputNeuron *i1 = [[InputNeuron alloc] initWithValue:.65];
    InputNeuron *i2 = [[InputNeuron alloc] initWithValue:.65];
    InputNeuron *i3 = [[InputNeuron alloc] initWithValue:.65];
    ReluNeuron *hidden1 = [[ReluNeuron alloc] init];
    ReluNeuron *hidden2 = [[ReluNeuron alloc] init];
    ReluNeuron *hidden3 = [[ReluNeuron alloc] init];
    ReluNeuron *hidden4 = [[ReluNeuron alloc] init];
    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];

    [hidden1 addInput:i1 withWeight:randF()];
    [hidden1 addInput:i2 withWeight:randF()];
    [hidden1 addInput:i3 withWeight:randF()];

    [hidden2 addInput:i1 withWeight:randF()];
    [hidden2 addInput:i2 withWeight:randF()];
    [hidden2 addInput:i3 withWeight:randF()];

    [hidden3 addInput:i1 withWeight:randF()];
    [hidden3 addInput:i2 withWeight:randF()];
    [hidden3 addInput:i3 withWeight:randF()];

    [hidden4 addInput:i1 withWeight:randF()];
    [hidden4 addInput:i2 withWeight:randF()];
    [hidden4 addInput:i3 withWeight:randF()];

    [output addInput:hidden1 withWeight:randF()];
    [output addInput:hidden2 withWeight:randF()];
    [output addInput:hidden3 withWeight:randF()];
    [output addInput:hidden4 withWeight:randF()];

    const CGFloat alpha = 0.2;

    // run the neural net
    void (^forwardPass)(void) = ^{
        [hidden1 forwardPass];
        [hidden2 forwardPass];
        [hidden3 forwardPass];
        [hidden4 forwardPass];
        [output forwardPass];
    };

    void (^backwardPass)(CGFloat) = ^(CGFloat goal) {
        [output backpropagateFor:[error errorDerivativeFor:goal forNeuron:output]];
        [hidden1 backpropagate];
        [hidden2 backpropagate];
        [hidden3 backpropagate];
        [hidden4 backpropagate];
        [output updateWeightsWithAlpha:alpha];
        [hidden1 updateWeightsWithAlpha:alpha];
        [hidden2 updateWeightsWithAlpha:alpha];
        [hidden3 updateWeightsWithAlpha:alpha];
        [hidden4 updateWeightsWithAlpha:alpha];
    };

    NSArray *data = @[@[@1, @0, @1],
                      @[@0, @1, @1],
                      @[@0, @0, @1],
                      @[@1, @1, @1]];


    NSArray *goal = @[@1, @1, @0, @0];

    // run the neural net

    CGFloat avgError = 0;

    for (NSInteger iter = 0; iter < 60; iter++) {
        for (NSInteger index = 0; index < [data count]; index++) {
            NSArray *testCase = data[index];
            NSNumber *target = goal[index];

            [i1 setActivation:[testCase[0] doubleValue]];
            [i2 setActivation:[testCase[1] doubleValue]];
            [i3 setActivation:[testCase[2] doubleValue]];

            forwardPass();
            backwardPass([target doubleValue]);

            avgError = avgError * 0.9 + ABS([error errorFor:[target doubleValue] forNeuron:output]) * 0.1;
        }
    }

    XCTAssertEqualWithAccuracy(avgError, 0, .01);

    for (NSInteger i = 0; i < [data count]; i++) {
        NSInteger index = i % [data count];
        NSArray *testCase = data[index];
        NSNumber *target = goal[index];

        [i1 setActivation:[testCase[0] doubleValue]];
        [i2 setActivation:[testCase[1] doubleValue]];
        [i3 setActivation:[testCase[2] doubleValue]];

        forwardPass();

        XCTAssertEqualWithAccuracy([output activation], [target doubleValue], .1);
    }
}

- (void)testLinearSeparator
{
    // hand set weights so that the error is exactly 0 all the time.
    ErrorCalculator *error = [[TraskMeanSquaredError alloc] init];
    InputNeuron *i1 = [[InputNeuron alloc] initWithValue:.65];
    InputNeuron *i2 = [[InputNeuron alloc] initWithValue:.65];
    StaticNeuron *b = [[StaticNeuron alloc] initWithValue:1.0];
    WeightedSumNeuron *h1 = [[WeightedSumNeuron alloc] init];
    WeightedSumNeuron *h2 = [[WeightedSumNeuron alloc] init];
    ClampNeuron *output = [[ClampNeuron alloc] initWithNeuron:[[WeightedSumNeuron alloc] init] andMin:0 andMax:1];

    [h1 addInput:i1 withWeight:1];
    [h1 addInput:i2 withWeight:0];
    [h1 addInput:b withWeight:8];

    [h2 addInput:i1 withWeight:0];
    [h2 addInput:i2 withWeight:2.0];
    [h2 addInput:b withWeight:0];

    [output addInput:h1 withWeight:-1.0];
    [output addInput:h2 withWeight:1.0];
    [output addInput:b withWeight:0];

    const CGFloat alpha = 0.001;
    __block CGFloat avgError = 0;

    // run the neural net
    void (^forwardPass)(void) = ^{
        [h1 forwardPass];
        [h2 forwardPass];
        [output forwardPass];
    };

    void (^backwardPass)(CGFloat) = ^(CGFloat goal) {
        if ((goal >= 1 && [output activation] < 1.0) ||
            (goal<1 && [output activation]> 0)) {
            [output backpropagateFor:[error errorDerivativeFor:goal forNeuron:output]];
            [h1 backpropagate];
            [h2 backpropagate];
            [output updateWeightsWithAlpha:alpha];
            [h1 updateWeightsWithAlpha:alpha];
            [h2 updateWeightsWithAlpha:alpha];
        }
    };

    // run the neural net

    CGFloat (^line)(CGFloat x) = ^(CGFloat x) {
        return 0.5 * x + 4;
    };

    for (NSInteger iter = 0; iter < 100000; iter++) {
        CGFloat x = rand() % 20 - 10;
        CGFloat y = rand() % 20 - 10;
        CGFloat g = y > line(x) ? 1 : 0;

        [i1 setActivation:x];
        [i2 setActivation:y];

        forwardPass();
        backwardPass(g);

        avgError = avgError * 0.9 + ABS([error errorFor:g forNeuron:output]) * 0.1;
    }

    XCTAssertEqualWithAccuracy(avgError, 0, .01);

    for (NSInteger i = 0; i < 10; i++) {
        CGFloat x = rand() % 20 - 10;
        CGFloat y = rand() % 20 - 10;
        CGFloat g = y > line(x) ? 1 : 0;

        [i1 setActivation:x];
        [i2 setActivation:y];

        forwardPass();

        XCTAssertEqualWithAccuracy([output activation], g, .1);
    }
}

- (void)testLinearSeparator2
{
    ErrorCalculator *error = [[TraskMeanSquaredError alloc] init];
    InputNeuron *i1 = [[InputNeuron alloc] initWithValue:.65];
    InputNeuron *i2 = [[InputNeuron alloc] initWithValue:.65];
    StaticNeuron *b = [[StaticNeuron alloc] initWithValue:1.0];
    WeightedSumNeuron *h1 = [[WeightedSumNeuron alloc] init];
    WeightedSumNeuron *h2 = [[WeightedSumNeuron alloc] init];
    ClampNeuron *output = [[ClampNeuron alloc] initWithNeuron:[[WeightedSumNeuron alloc] init] andMin:0 andMax:1];

    [h1 addInput:i1 withWeight:.1];
    [h1 addInput:i2 withWeight:.2];
    [h1 addInput:b withWeight:.3];

    [h2 addInput:i1 withWeight:.4];
    [h2 addInput:i2 withWeight:.5];
    [h2 addInput:b withWeight:.6];

    [output addInput:h1 withWeight:0.15];
    [output addInput:h2 withWeight:0.25];
    [output addInput:b withWeight:0.35];

    const CGFloat alpha = 0.01;
    __block CGFloat avgError = 0;

    // run the neural net
    void (^forwardPass)(void) = ^{
        [h1 forwardPass];
        [h2 forwardPass];
        [output forwardPass];
    };

    void (^backwardPass)(CGFloat) = ^(CGFloat goal) {
        if ((goal >= 1 && [output activation] < 1.0) ||
            (goal<1 && [output activation]> 0)) {
            [output backpropagateFor:[error errorDerivativeFor:goal forNeuron:output]];
            [h1 backpropagate];
            [h2 backpropagate];
            [output updateWeightsWithAlpha:alpha];
            [h1 updateWeightsWithAlpha:alpha];
            [h2 updateWeightsWithAlpha:alpha];
        }
    };

    // run the neural net

    // here's our divider line that we'll use to train
    CGFloat (^line)(CGFloat x) = ^(CGFloat x) {
        return 0.5 * x + 4;
    };

    for (NSInteger iter = 0; iter < 100000; iter++) {
        // pick out random points between (-10,-10) and (10,10)
        // g is 1 if the point is above the line, 0 otherwise
        CGFloat x = rand() % 20 - 10;
        CGFloat y = rand() % 20 - 10;
        CGFloat g = y > line(x) ? 1 : 0;

        [i1 setActivation:x];
        [i2 setActivation:y];

        forwardPass();
        backwardPass(g);

        avgError = avgError * 0.9 + ABS([error errorFor:g forNeuron:output]) * 0.1;
    }

    // at this point, our network can predict if the input point
    // is above or below our line.
    XCTAssertEqualWithAccuracy(avgError, 0, .01);

    for (NSInteger i = 0; i < 10; i++) {
        CGFloat x = rand() % 20 - 10;
        CGFloat y = rand() % 20 - 10;
        CGFloat g = y > line(x) ? 1 : 0;

        [i1 setActivation:x];
        [i2 setActivation:y];

        forwardPass();

        XCTAssertEqualWithAccuracy([output activation], g, .1);
    }
}

- (void)testLinearSeparator3
{
    ErrorCalculator *error = [[TraskMeanSquaredError alloc] init];
    InputNeuron *i1 = [[InputNeuron alloc] initWithValue:.65];
    InputNeuron *i2 = [[InputNeuron alloc] initWithValue:.65];
    StaticNeuron *b = [[StaticNeuron alloc] initWithValue:1.0];
    ClampNeuron *output = [[ClampNeuron alloc] initWithNeuron:[[WeightedSumNeuron alloc] init] andMin:0 andMax:1];

    [output addInput:i1 withWeight:.1];
    [output addInput:i2 withWeight:.2];
    [output addInput:b withWeight:.3];

    const CGFloat alpha = 0.01;
    __block CGFloat avgError = 0;

    // run the neural net
    void (^forwardPass)(void) = ^{
        [output forwardPass];
    };

    void (^backwardPass)(CGFloat) = ^(CGFloat goal) {
        if ((goal >= 1 && [output activation] < 1.0) ||
            (goal<1 && [output activation]> 0)) {
            [output backpropagateFor:[error errorDerivativeFor:goal forNeuron:output]];
            [output updateWeightsWithAlpha:alpha];
        }
    };

    // run the neural net

    // here's our divider line that we'll use to train
    CGFloat (^line)(CGFloat x) = ^(CGFloat x) {
        return 0.5 * x + 4;
    };

    for (NSInteger iter = 0; iter < 100000; iter++) {
        // pick out random points between (-10,-10) and (10,10)
        // g is 1 if the point is above the line, 0 otherwise
        CGFloat x = rand() % 20 - 10;
        CGFloat y = rand() % 20 - 10;
        CGFloat g = y > line(x) ? 1 : 0;

        [i1 setActivation:x];
        [i2 setActivation:y];

        forwardPass();
        backwardPass(g);

        avgError = avgError * 0.9 + ABS([error errorFor:g forNeuron:output]) * 0.1;
    }

    // at this point, our network can predict if the input point
    // is above or below our line.
    XCTAssertEqualWithAccuracy(avgError, 0, .01);

    for (NSInteger i = 0; i < 10; i++) {
        CGFloat x = rand() % 20 - 10;
        CGFloat y = rand() % 20 - 10;
        CGFloat g = y > line(x) ? 1 : 0;

        [i1 setActivation:x];
        [i2 setActivation:y];

        forwardPass();

        XCTAssertEqualWithAccuracy([output activation], g, .1);
    }
}

- (void)testSimpleLinear
{
    ErrorCalculator *error = [[TraskMeanSquaredError alloc] init];
    InputNeuron *i1 = [[InputNeuron alloc] initWithValue:.65];
    StaticNeuron *b = [[StaticNeuron alloc] initWithValue:1.0];
    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];

    [output addInput:i1 withWeight:.01];
    [output addInput:b withWeight:.03];

    const CGFloat alpha = 0.04;
    __block CGFloat avgError = 0;

    // run the neural net
    void (^forwardPass)(void) = ^{
        [output forwardPass];
    };

    void (^backwardPass)(CGFloat) = ^(CGFloat goal) {
        if ((goal >= 1 && [output activation] < 1.0) ||
            (goal<1 && [output activation]> 0)) {
            [output backpropagateFor:[error errorDerivativeFor:goal forNeuron:output]];
            [output updateWeightsWithAlpha:alpha];
        }
    };

    // run the neural net

    // here's our divider line that we'll use to train
    CGFloat (^line)(CGFloat x) = ^(CGFloat x) {
        return 0.5 * x + 4;
    };

    for (NSInteger iter = 0; iter < 5000; iter++) {
        // pick out random points between (-10,-10) and (10,10)
        // g is 1 if the point is above the line, 0 otherwise
        CGFloat x = rand() % 20 - 10;
        CGFloat y = line(x);

        [i1 setActivation:x];

        forwardPass();
        backwardPass(y);

        avgError = avgError * 0.9 + ABS([error errorFor:y forNeuron:output]) * 0.1;
    }

    // at this point, our network can predict if the input point
    // is above or below our line.
    XCTAssertEqualWithAccuracy(avgError, 0, .01);

    for (NSInteger i = 0; i < 10; i++) {
        CGFloat x = rand() % 20 - 10;
        CGFloat y = line(x);

        [i1 setActivation:x];

        forwardPass();

        XCTAssertEqualWithAccuracy([output activation], y, .1);
    }
}

- (void)testSimpleLinear2
{
    ErrorCalculator *error = [ABSError calculator];
    InputNeuron *i1 = [[InputNeuron alloc] initWithValue:.65];
    StaticNeuron *b = [[StaticNeuron alloc] initWithValue:1.0];
    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];

    [output addInput:i1 withWeight:.1];
    [output addInput:b withWeight:.2];

    const CGFloat alpha = 0.01;
    __block CGFloat avgError = 0;

    // run the neural net
    void (^forwardPass)(void) = ^{
        [output forwardPass];
    };

    void (^backwardPass)(CGFloat) = ^(CGFloat goal) {
        [output backpropagateFor:[error errorDerivativeFor:goal forNeuron:output]];
        [output updateWeightsWithAlpha:alpha];
    };

    // run the neural net

    // here's our divider line that we'll use to train
    CGFloat (^line)(CGFloat x) = ^(CGFloat x) {
        return 1.8 * x + 32;
    };

    for (NSInteger iter = 0; iter < 5000; iter++) {
        // pick out random points between (-10,-10) and (10,10)
        // g is 1 if the point is above the line, 0 otherwise
        CGFloat x = rand() % 20 - 10;
        CGFloat y = line(x);

        [i1 setActivation:x];

        forwardPass();
        backwardPass(y);

        avgError = avgError * 0.9 + ABS([error errorFor:y forNeuron:output]) * 0.1;
    }

    // at this point, our network can predict if the input point
    // is above or below our line.
    XCTAssertEqualWithAccuracy(avgError, 0, .25);

    for (NSInteger i = 0; i < 10; i++) {
        CGFloat x = rand() % 20 - 10;
        CGFloat y = line(x);

        [i1 setActivation:x];

        forwardPass();

        XCTAssertEqualWithAccuracy([output activation], y, .25);
    }
}

- (void)testKerasSimpleLinear
{
    ErrorCalculator *error = [MeanSquaredError calculator];
    InputNeuron *i1 = [[InputNeuron alloc] initWithValue:1];
    InputNeuron *i2 = [[InputNeuron alloc] initWithValue:0];
    InputNeuron *i3 = [[InputNeuron alloc] initWithValue:1];

    WeightedSumNeuron *hidden1 = [[WeightedSumNeuron alloc] init];
    WeightedSumNeuron *hidden2 = [[WeightedSumNeuron alloc] init];
    WeightedSumNeuron *hidden3 = [[WeightedSumNeuron alloc] init];

    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];

    [hidden1 addInput:i1 withWeight:0.17318463];
    [hidden1 addInput:i2 withWeight:-0.01949477];
    [hidden1 addInput:i3 withWeight:-0.67797828];

    [hidden2 addInput:i1 withWeight:0.50842547];
    [hidden2 addInput:i2 withWeight:0.5068841];
    [hidden2 addInput:i3 withWeight:0.37353373];

    [hidden3 addInput:i1 withWeight:0.18504167];
    [hidden3 addInput:i2 withWeight:-0.58431745];
    [hidden3 addInput:i3 withWeight:-0.51648164];

    [output addInput:hidden1 withWeight:0.35634112];
    [output addInput:hidden2 withWeight:-0.69503939];
    [output addInput:hidden3 withWeight:-1.20925915];

    // why is our alpha twice that of tensorflow?
    const CGFloat alpha = 0.1;

    // run the neural net
    void (^forwardPass)(void) = ^{
        [hidden1 forwardPass];
        [hidden2 forwardPass];
        [hidden3 forwardPass];
        [output forwardPass];
    };

    void (^backwardPass)(CGFloat) = ^(CGFloat goal) {
        [output backpropagateFor:[error errorDerivativeFor:goal forNeuron:output]];
        [hidden1 backpropagate];
        [hidden2 backpropagate];
        [hidden3 backpropagate];
        [output updateWeightsWithAlpha:alpha];
        [hidden1 updateWeightsWithAlpha:alpha];
        [hidden2 updateWeightsWithAlpha:alpha];
        [hidden3 updateWeightsWithAlpha:alpha];
    };

    NSArray *data = @[@[@1, @0, @1]];
    NSArray *goal = @[@1];

    // run the neural net
    [i1 setActivation:[data[0][0] doubleValue]];
    [i2 setActivation:[data[0][1] doubleValue]];
    [i3 setActivation:[data[0][2] doubleValue]];

    forwardPass();

    // before we backpropagate, check that our prediction with these weights matches that from Tensorflow
    XCTAssertEqualWithAccuracy(ABS([error errorFor:1.0 forNeuron:output]), 1.93788230419, .000001);
    XCTAssertEqualWithAccuracy([output activation], -0.39207834, .000001);

    backwardPass([goal[0] doubleValue]);

    // after we've backpropagated once, check that our new weights match Tensorflow
    XCTAssertEqualWithAccuracy([hidden1.weights[0] doubleValue], 0.27239558, .000001);
    XCTAssertEqualWithAccuracy([hidden1.weights[1] doubleValue], -0.01949477, .000001);
    XCTAssertEqualWithAccuracy([hidden1.weights[2] doubleValue], -0.5787673, .000001);

    XCTAssertEqualWithAccuracy([hidden2.weights[0] doubleValue], 0.3149156, .000001);
    XCTAssertEqualWithAccuracy([hidden2.weights[1] doubleValue], 0.5068841, .000001);
    XCTAssertEqualWithAccuracy([hidden2.weights[2] doubleValue], 0.18002386, .000001);

    XCTAssertEqualWithAccuracy([hidden3.weights[0] doubleValue], -0.15163505, .000001);
    XCTAssertEqualWithAccuracy([hidden3.weights[1] doubleValue], -0.58431745, .000001);
    XCTAssertEqualWithAccuracy([hidden3.weights[2] doubleValue], -0.85315835, .000001);

    XCTAssertEqualWithAccuracy([output.weights[0] doubleValue], 0.21579865, .000001);
    XCTAssertEqualWithAccuracy([output.weights[1] doubleValue], -0.4494881, .000001);
    XCTAssertEqualWithAccuracy([output.weights[2] doubleValue], -1.30153728, .000001);

    forwardPass();

    // after we've backpropagated once, check that our new prediction and error match Tensorflow
    XCTAssertEqualWithAccuracy(ABS([error errorFor:1.0 forNeuron:output]), 0.000368336681277, .000001);
    XCTAssertEqualWithAccuracy([output activation], 1.0191921, .000001);
}


- (void)testKerasSimpleLinear2
{
    ErrorCalculator *error = [MeanSquaredError calculator];
    InputNeuron *i1 = [[InputNeuron alloc] initWithValue:2.1];
    InputNeuron *i2 = [[InputNeuron alloc] initWithValue:0.3];
    InputNeuron *i3 = [[InputNeuron alloc] initWithValue:3.1];

    WeightedSumNeuron *hidden1 = [[WeightedSumNeuron alloc] init];
    WeightedSumNeuron *hidden2 = [[WeightedSumNeuron alloc] init];
    WeightedSumNeuron *hidden3 = [[WeightedSumNeuron alloc] init];

    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];

    [hidden1 addInput:i1 withWeight:0.17318463];
    [hidden1 addInput:i2 withWeight:-0.01949477];
    [hidden1 addInput:i3 withWeight:-0.67797828];

    [hidden2 addInput:i1 withWeight:0.50842547];
    [hidden2 addInput:i2 withWeight:0.5068841];
    [hidden2 addInput:i3 withWeight:0.37353373];

    [hidden3 addInput:i1 withWeight:0.18504167];
    [hidden3 addInput:i2 withWeight:-0.58431745];
    [hidden3 addInput:i3 withWeight:-0.51648164];

    [output addInput:hidden1 withWeight:0.35634112];
    [output addInput:hidden2 withWeight:-0.69503939];
    [output addInput:hidden3 withWeight:-1.20925915];

    // why is our alpha twice that of tensorflow?
    const CGFloat alpha = 0.1;

    // run the neural net
    void (^forwardPass)(void) = ^{
        [hidden1 forwardPass];
        [hidden2 forwardPass];
        [hidden3 forwardPass];
        [output forwardPass];
    };

    void (^backwardPass)(CGFloat) = ^(CGFloat goal) {
        [output backpropagateFor:[error errorDerivativeFor:goal forNeuron:output]];
        [hidden1 backpropagate];
        [hidden2 backpropagate];
        [hidden3 backpropagate];
        [output updateWeightsWithAlpha:alpha];
        [hidden1 updateWeightsWithAlpha:alpha];
        [hidden2 updateWeightsWithAlpha:alpha];
        [hidden3 updateWeightsWithAlpha:alpha];
    };

    NSArray *data = @[@[@2.1, @0.3, @3.1]];
    NSArray *goal = @[@0.7];

    // run the neural net
    [i1 setActivation:[data[0][0] doubleValue]];
    [i2 setActivation:[data[0][1] doubleValue]];
    [i3 setActivation:[data[0][2] doubleValue]];

    forwardPass();

    // before we backpropagate, check that our prediction with these weights matches that from Tensorflow
    XCTAssertEqualWithAccuracy(ABS([error errorFor:[goal[0] doubleValue] forNeuron:output]), 1.67913460732, .000001);
    XCTAssertEqualWithAccuracy([output activation], -0.59581435, .000001);

    backwardPass([goal[0] doubleValue]);

    // after we've backpropagated once, check that our new weights match Tensorflow
    XCTAssertEqualWithAccuracy([hidden1.weights[0] doubleValue], 0.36712044, .000001);
    XCTAssertEqualWithAccuracy([hidden1.weights[1] doubleValue], 0.00821034, .000001);
    XCTAssertEqualWithAccuracy([hidden1.weights[2] doubleValue], -0.3916921, .000001);

    XCTAssertEqualWithAccuracy([hidden2.weights[0] doubleValue], 0.13015583, .000001);
    XCTAssertEqualWithAccuracy([hidden2.weights[1] doubleValue], 0.45284557, .000001);
    XCTAssertEqualWithAccuracy([hidden2.weights[2] doubleValue], -0.18486428, .000001);

    XCTAssertEqualWithAccuracy([hidden3.weights[0] doubleValue], -0.47308791, .000001);
    XCTAssertEqualWithAccuracy([hidden3.weights[1] doubleValue], -0.67833596, .000001);
    XCTAssertEqualWithAccuracy([hidden3.weights[2] doubleValue], -1.48800635, .000001);

    XCTAssertEqualWithAccuracy([output.weights[0] doubleValue], -0.09561118, .000001);
    XCTAssertEqualWithAccuracy([output.weights[1] doubleValue], -0.07882446, .000001);
    XCTAssertEqualWithAccuracy([output.weights[2] doubleValue], -1.56892562, .000001);

    forwardPass();

    // after we've backpropagated once, check that our new prediction and error match Tensorflow
    XCTAssertEqualWithAccuracy(ABS([error errorFor:[goal[0] doubleValue] forNeuron:output]), 71.7446060, .0001);
    XCTAssertEqualWithAccuracy([output activation], 9.17021847, .00001);
}

- (void)testChapter1
{
    ErrorCalculator *error = [ABSError calculator];
    InputNeuron *i1 = [[InputNeuron alloc] initWithValue:.65];
    StaticNeuron *b = [[StaticNeuron alloc] initWithValue:1.0];
    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];

    [output addInput:i1 withWeight:.1];
    [output addInput:b withWeight:.2];

    const CGFloat alpha = 0.01;
    __block CGFloat avgError = 0;

    // run the neural net
    void (^forwardPass)(void) = ^{
        [output forwardPass];
    };

    void (^backwardPass)(CGFloat) = ^(CGFloat goal) {
        [output backpropagateFor:[error errorDerivativeFor:goal forNeuron:output]];
        [output updateWeightsWithAlpha:alpha];
    };

    // run the neural net

    // here's our divider line that we'll use to train
    CGFloat (^line)(CGFloat x) = ^(CGFloat x) {
        return 1.8 * x + 32;
    };

    for (NSInteger iter = 0; iter < 3600; iter++) {
        // pick out random points between (-10,-10) and (10,10)
        // g is 1 if the point is above the line, 0 otherwise
        CGFloat x = rand() % 20 - 10;
        CGFloat y = line(x);

        [i1 setActivation:x];

        forwardPass();
        backwardPass(y);

        avgError = avgError * 0.99 + ABS([error errorFor:y forNeuron:output]) * 0.01;
    }

    // at this point, our network can predict if the input point
    // is above or below our line.
    XCTAssertEqualWithAccuracy(avgError, 0, .25);
}

- (void)testChapter2
{
    ErrorCalculator *error = [MeanSquaredError calculator];
    InputNeuron *i1 = [[InputNeuron alloc] initWithValue:.65];
    StaticNeuron *b = [[StaticNeuron alloc] initWithValue:1.0];
    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];

    [output addInput:i1 withWeight:.1];
    [output addInput:b withWeight:.2];

    const CGFloat alpha = 0.01;
    __block CGFloat avgError = 0;

    // run the neural net
    void (^forwardPass)(void) = ^{
        [output forwardPass];
    };

    void (^backwardPass)(CGFloat) = ^(CGFloat goal) {
        [output backpropagateFor:[error errorDerivativeFor:goal forNeuron:output]];
        [output updateWeightsWithAlpha:alpha];
    };

    // run the neural net

    // here's our divider line that we'll use to train
    CGFloat (^line)(CGFloat x) = ^(CGFloat x) {
        return 1.8 * x + 32;
    };

    for (NSInteger iter = 0; iter < 800; iter++) {
        // pick out random points between (-10,-10) and (10,10)
        // g is 1 if the point is above the line, 0 otherwise
        CGFloat x = rand() % 20 - 10;
        CGFloat y = line(x);

        [i1 setActivation:x];

        forwardPass();
        backwardPass(y);

        avgError = avgError * 0.99 + ABS([error errorFor:y forNeuron:output]) * 0.01;
    }

    // at this point, our network can predict if the input point
    // is above or below our line.
    XCTAssertEqualWithAccuracy(avgError, 0, .25);
}

@end

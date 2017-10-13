//
//  neuralnetTests.m
//  neuralnetTests
//
//  Created by Adam Wulf on 10/9/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import <XCTest/XCTest.h>
#import <neuralnet/neuralnet.h>

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
- (void)testChapter3Page34
{
    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
    StaticNeuron *i1 = [[StaticNeuron alloc] initWithValue:8.5];
    StaticNeuron *i2 = [[StaticNeuron alloc] initWithValue:.65];
    StaticNeuron *i3 = [[StaticNeuron alloc] initWithValue:1.2];
    WeightedSumNeuron *output = [[WeightedSumNeuron alloc] init];
    [output addInput:i1 withWeight:0.1];
    [output addInput:i2 withWeight:0.2];
    [output addInput:i3 withWeight:0];

    // run the neural net
    [output forwardPass];

    XCTAssertEqualWithAccuracy(.98, [output output], .0000000000001, @"prediction is correct");
}

// test case from Grokking Deep Learning book by Andrew W. Trask
- (void)testChapter3Page37
{
    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
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

    XCTAssertEqualWithAccuracy(.195, [output1 output], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(.13, [output2 output], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(.585, [output3 output], .0000000000001, @"prediction is correct");
}

// test case from Grokking Deep Learning book by Andrew W. Trask
- (void)testChapter3Page39
{
    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
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

    XCTAssertEqualWithAccuracy(.555, [output1 output], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(.98, [output2 output], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(.965, [output3 output], .0000000000001, @"prediction is correct");
}

// test case from Grokking Deep Learning book by Andrew W. Trask
- (void)testChapter3Page43
{
    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
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

    XCTAssertEqualWithAccuracy(.86, [hidden1 output], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(.295, [hidden2 output], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(1.23, [hidden3 output], .0000000000001, @"prediction is correct");

    // note, the book rounds outputs 1 and 3 to 3 digits
    XCTAssertEqualWithAccuracy(.2135, [output1 output], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(.145, [output2 output], .0000000000001, @"prediction is correct");
    XCTAssertEqualWithAccuracy(.5065, [output3 output], .0000000000001, @"prediction is correct");
}

@end

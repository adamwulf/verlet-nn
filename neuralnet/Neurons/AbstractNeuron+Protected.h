//
//  AbstractNeuron+Protected.h
//  neuralnet
//
//  Created by Adam Wulf on 10/9/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import "AbstractNeuron.h"

@interface AbstractNeuron ()

@property(nonatomic, readonly) NSMutableArray *inputs;
@property(nonatomic, readonly) NSMutableArray *outputs;
@property(nonatomic, readonly) NSMutableArray *weights;
@property(nonatomic, assign) CGFloat activation;

- (void)addOutput:(AbstractNeuron *)neuron;

#pragma mark - Abstract

// probably the sigmoid function
- (CGFloat)transferFunction:(CGFloat)activation;

// probably the derivative of the sigmoid function
- (CGFloat)transferDerivative;

- (void)updateWeightAtIndex:(NSInteger)index with:(CGFloat)weight;

@end

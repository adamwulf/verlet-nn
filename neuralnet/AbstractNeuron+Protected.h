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
@property(nonatomic, readonly) NSMutableArray *previousWeights;

@property(nonatomic, assign) CGFloat activation;
@property(nonatomic, readonly) CGFloat derivative;

- (CGFloat)derivativeInputAtIndex:(NSInteger)neuronIndex andGoal:(CGFloat)goal;

- (void)addOutput:(AbstractNeuron *)neuron;

@end

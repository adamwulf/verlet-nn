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
@property(nonatomic, readonly) NSMutableArray *weights;

@property(nonatomic, assign) CGFloat currentValue;

- (CGFloat)derivativeInputAtIndex:(NSInteger)neuronIndex andGoal:(CGFloat)goal;

@end

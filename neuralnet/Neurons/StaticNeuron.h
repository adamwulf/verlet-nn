//
//  StaticNeuron.h
//  neuralnet
//
//  Created by Adam Wulf on 10/9/17.
//  Copyright © 2017 Adam Wulf. All rights reserved.
//

#import "AbstractNeuron.h"

@interface StaticNeuron : AbstractNeuron

- (instancetype)initWithValue:(CGFloat)value;

@end

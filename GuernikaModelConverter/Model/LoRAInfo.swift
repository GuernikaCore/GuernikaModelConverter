//
//  LoRAInfo.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 11/8/23.
//

import Foundation

struct LoRAInfo: Hashable, Identifiable {
    var id: URL { url }
    var url: URL
    var ratio: Double
    
    var argument: String {
        String(format: "%@:%0.2f", url.path(percentEncoded: false), ratio)
    }
}

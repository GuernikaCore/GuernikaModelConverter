//
//  CircularProgress.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 15/12/22.
//

import SwiftUI

struct CircularProgress: View {
    var progress: Float?
    
    var body: some View {
        ZStack {
            if let progress, progress != 0 && progress != 1 {
                Circle()
                    .stroke(lineWidth: 4)
                    .opacity(0.2)
                    .foregroundColor(Color.primary)
                    .frame(width: 20, height: 20)
                Circle()
                    .trim(from: 0, to: CGFloat(min(progress, 1)))
                    .stroke(style: StrokeStyle(lineWidth: 4, lineCap: .round, lineJoin: .round))
                    .foregroundColor(Color.accentColor)
                    .rotationEffect(Angle(degrees: 270))
                    .animation(.linear, value: progress)
                    .frame(width: 20, height: 20)
            } else {
                ProgressView()
                    .progressViewStyle(.circular)
                    .scaleEffect(0.7)
            }
        }.frame(width: 24, height: 24)
    }
}

struct CircularProgress_Previews: PreviewProvider {
    static var previews: some View {
        CircularProgress(progress: 0.5)
    }
}

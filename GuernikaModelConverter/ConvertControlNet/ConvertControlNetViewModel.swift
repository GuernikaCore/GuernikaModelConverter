//
//  ConvertControlNetViewModel.swift
//  GuernikaModelConverter
//
//  Created by Guillermo Cique Fernández on 30/3/23.
//

import SwiftUI
import Combine

class ConvertControlNetViewModel: ObservableObject {
    @Published var showSuccess: Bool = false
    var showError: Bool {
        get { error != nil }
        set {
            if !newValue {
                error = nil
                isCoreMLError = false
            }
        }
    }
    var isCoreMLError: Bool = false
    @Published var error: String?
    
    var isReady: Bool {
        switch controlNetOrigin {
        case .huggingface: return !huggingfaceIdentifier.isEmpty
        case .diffusers: return diffusersLocation != nil
        case .checkpoint: return checkpointLocation != nil
        }
    }
    @Published var process: ConverterProcess?
    var isRunning: Bool { process != nil }
    
    @Published var showOutputPicker: Bool = false
    @AppStorage("output_location") var outputLocation: URL?
    
    var controlNetOrigin: ModelOrigin {
        get { ModelOrigin(rawValue: controlNetOriginString) ?? .huggingface }
        set { controlNetOriginString = newValue.rawValue }
    }
    @AppStorage("controlnet_origin") var controlNetOriginString: String = ModelOrigin.huggingface.rawValue
    @AppStorage("controlnet_huggingface_identifier") var huggingfaceIdentifier: String = ""
    @AppStorage("controlnet_diffusers_location") var diffusersLocation: URL?
    @AppStorage("controlnet_checkpoint_location") var checkpointLocation: URL?
    var selectedControlNet: String? {
        switch controlNetOrigin {
        case .huggingface: return nil
        case .diffusers: return diffusersLocation?.lastPathComponent
        case .checkpoint: return checkpointLocation?.lastPathComponent
        }
    }
    
    var computeUnits: ComputeUnits {
        get { ComputeUnits(rawValue: computeUnitsString) ?? .cpuAndNeuralEngine }
        set { computeUnitsString = newValue.rawValue }
    }
    @AppStorage("compute_units") var computeUnitsString: String = ComputeUnits.cpuAndNeuralEngine.rawValue
    
    @AppStorage("custom_size") var customSize: Bool = false
    @AppStorage("custom_size_width") var customWidth: Int = 512
    @AppStorage("custom_size_height") var customHeight: Int = 512
    
    private var cancellables: Set<AnyCancellable> = []
    
    func start() {
        guard checkCoreMLCompiler() else {
            isCoreMLError = true
            error = "CoreMLCompiler not available.\nMake sure you have Xcode installed and you run \"sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer/\" on a Terminal."
            return
        }
        do {
            let process = try ConverterProcess(
                outputLocation: outputLocation,
                controlNetOrigin: controlNetOrigin,
                huggingfaceIdentifier: huggingfaceIdentifier,
                diffusersLocation: diffusersLocation,
                checkpointLocation: checkpointLocation,
                computeUnits: computeUnits,
                customWidth: customSize && computeUnits == .cpuAndGPU ? customWidth : nil,
                customHeight: customSize && computeUnits == .cpuAndGPU ? customHeight : nil
            )
            process.objectWillChange
                .receive(on: DispatchQueue.main)
                .sink { _ in
                    self.objectWillChange.send()
                }.store(in: &cancellables)
            NotificationCenter.default.publisher(for: Process.didTerminateNotification, object: process.process)
                .receive(on: DispatchQueue.main)
                .sink { _ in
                    withAnimation {
                        if process.didComplete {
                            self.showSuccess = true
                        }
                        self.cancel()
                    }
                }.store(in: &cancellables)
            try process.start()
            withAnimation {
                self.process = process
            }
        } catch ConverterProcess.ArgumentError.noOutputLocation {
            showOutputPicker = true
        } catch ConverterProcess.ArgumentError.noHuggingfaceIdentifier {
            self.error = "Enter a valid identifier"
        } catch ConverterProcess.ArgumentError.noDiffusersLocation {
            self.error = "Enter a valid location"
        } catch ConverterProcess.ArgumentError.noCheckpointLocation {
            self.error = "Enter a valid location"
        } catch {
            print(error.localizedDescription)
            self.error = error.localizedDescription
            withAnimation {
                cancel()
            }
        }
    }
    
    func cancel() {
        withAnimation {
            process?.cancel()
            process = nil
        }
    }
    
    func checkCoreMLCompiler() -> Bool {
        let process = Process()
        process.executableURL = URL(string: "file:///usr/bin/xcrun")
        process.arguments = ["coremlcompiler", "version"]
        do {
            try process.run()
        } catch {
            Logger.shared.append("coremlcompiler not found")
            return false
        }
        return true
    }
}

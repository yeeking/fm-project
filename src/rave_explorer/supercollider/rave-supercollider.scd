NN.deQuarantine;
s.reboot;

(
// 1. load
s.waitForBoot {
	var model_file, audio_file;
    // when in a Routine, this method waits until the model has loaded
	model_file = "../FM_Project/fm-project/exported_models/fm-4000-run-3_streaming.ts";
	NN.load(\rave4000, model_file);
    // when model has loaded, print a description of all methods and attributes
         NN(\rave4000).describe;
   // load a buffer 
//	audio_file = "../fm_project/audio/flute_short_notes-001.wav";
//	~b1 = Buffer.read(s, audio_file);
}
)

(
SynthDef(\bufferToBus, {
    |bufnum = 0, outbus = 0, amp = 1.0|
    var sig;

    // Play buffer once, no looping
    sig = PlayBuf.ar(
        numChannels: 1, // Adjust if your file is stereo
        bufnum: bufnum,
        rate: BufRateScale.kr(bufnum),
		loop: 1
		// doneAction: 2 // free synth when done
    );

    // Send to internal bus
	// Out.ar([outbus, 0, 1], sig * amp);
	Out.ar(outbus, sig * MouseX.kr(0.1, 2.0));
}).add;

SynthDef(\bus_to_rave, {
	| busnum, amp = 1.0|
	var in_sig, sig;
	in_sig = In.ar(busnum, 1);
	sig = NN(\rave4000, \forward).ar(in_sig.dup);
	Out.ar(0, in_sig*MouseY.kr() + sig*amp);
}).add;

SynthDef(\mic_to_rave_1, {
	| amp = 1.0|
	var in_sig, sig;
	in_sig = SoundIn.ar([0,1]).mean*MouseX.kr(0.001, 2.0);
	sig = NN(\rave4000, \forward).ar(in_sig);
	Out.ar([0, 1],sig);
}).add;


SynthDef(\mic_to_rave_2, {
	| amp = 1.0|
	var in_sig, sig;
	in_sig = SoundIn.ar(0).dup*2.0;
	sig = NN(\rave4000, \forward).ar(in_sig);
	Out.ar([0, 1],sig);
}).add;


)



// input 0 -> model . Mouse X is input gain, Mouse Y is input pass through
(
~vsynth3 = Synth(\mic_to_rave_2, []);
)

// play audio buffer to model≥
(
~bus1 = Bus.audio(s, 1); // 1-channel audio bus

~synth2 = Synth(\bus_to_rave, [
    \busnum,  ~bus1.index,
]);
~synth1 = Synth(\bufferToBus, [
    \bufnum, ~b1.bufnum,
    \outbus, ~bus1.index
]);
)



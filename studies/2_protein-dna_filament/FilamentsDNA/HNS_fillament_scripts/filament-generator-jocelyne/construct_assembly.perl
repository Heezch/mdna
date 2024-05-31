#!/usr/bin/perl

#script for constructing an H-NS filament
#Jocelyne Vreede 2014

my $monomers = 10;
if (@ARGV) {$monomers = shift(@ARGV);}
if ($monomers < 2) {print "Cannot make filament with less than 2 monomers\n";
		    $monomers = 2;}
print "Constructing filament with $monomers monomers\n";

#directories containing pdb files used as building blocks
my @structure;
$structure[0] = "site1";
$structure[1] = "helix3";
$structure[2] = "site2";
$structure[3] = "linker";
$structure[4] = "dbd";
$structure[5] = "dna";

#residue ranges for the different domains
#site1: contains two chains, Y and Z, from residue 1 to 64
#helix3: containes one chain, Y, from residue 22 to 64
#site2: contains two chains, Y and Z, from residue 55 to 83, and DNA, chains 1 and 2
#linker: contains one chain, Y, from residue 71 to 100, and DNA, chains 1 and 2
#dbd: contains one chain, Y and Z, from residue 64 to 137, and DNA, chains 1 and 2
#dna: contains three chains, Y, 1 and 2, from residue 112 to 114, and DNA, chains 1 and 2

my %overlap;
$overlap{"site1-helix3"} = "22-40";
$overlap{"helix3-site1"} = "22-40";
$overlap{"helix3-site2"} = "55-64";
$overlap{"site2-helix3"} = "55-64";
$overlap{"site2-linker"} = "71-83";
$overlap{"linker-site2"} = "71-83";
$overlap{"linker-dbd"}   = "96-100";  
$overlap{"dbd-linker"}   = "96-100";  
$overlap{"dbd-dna"}      = "112-114";
$overlap{"dna-dbd"}      = "112-114";
 
#global declaration of variables
my @assembly = ();
my @fit = ();
my @buffer = ();
my @chainsequence = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
		     "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z");
my $currchains = 0;

open(LOG, ">log");

#INITIALIZATION

print "Initializing .. ";
print LOG "Start with site2 ";
my $structurefile = &ChooseRandomStructure("site2");
print LOG "using $structurefile";
push(@assembly, [ &AddPdb($structurefile, "Y") ]);
$currchains = 0;

#rename chains, will start from A
&RenameChainsAssembly($currchains);

#write current assembly to file
my $currfile = &WritePdb($currchains, 2);
#print "written $currfile\n";

print "Initialization finished\n";

print "Constructing assembly .. ";
# 0-site1; 1-helix3; 2-site2; 3-linker; 4-dbd; 5-dna
my @SequenceFromsite1 = (1,2,3,4);
my @SequenceFromsite2N = (1,0);
my @SequenceFromsite2C = (3,4);

#loop until all chains are added
for (my $c = $currchains; $c < $monomers ; $c++) {
  &AddSequence(0, 0, @SequenceFromsite2C);   #no reversal: going from N to C; no new chain
  &AddSequence(1, 1, @SequenceFromsite2N);   #reversal: going from C to N; new chain
  $c++;
  $currchains++;
  $currfile = &WritePdb($currchains, 0);
  &AddSequence(0, 1, @SequenceFromsite1);   #no reversal: going from N to C; new chain
  $currchains++;
  $currfile = &WritePdb($currchains, 0);
}

#print to file
$currfile = WritePdbFinal($monomers);
print "Construction finished .. $currfile\n";
&EM($currfile);
close(LOG);

sub EM {
my $file = shift(@_);
open(PDB2GMX, "|pdb2gmx -f $file >& pdb2gmx.log");
print PDB2GMX "6\n";
print PDB2GMX "1\n";
close(PDB2GMX);

`editconf -f conf.gro -bt cubic -d 3 -o boxed.gro >& editconf.log`;
open(MDP, ">em.mdp"); 
print MDP "define                   = -DFLEXIBLE\n";
print MDP "integrator               = cg\n";
print MDP "nsteps                   = 1000000\n";
print MDP "emtol                    = 10000\n";
print MDP "emstep                   = 0.01\n";
print MDP "niter                    = 20\n";
print MDP "fcstep                   = 0\n";
print MDP "nstcgsteep               = 100\n";
print MDP "nbfgscorr                = 10\n";
print MDP "nstxout                  = 0\n";
print MDP "nstvout                  = 0\n";
print MDP "nstfout                  = 0\n";
print MDP "nstlog                   = 250\n";
print MDP "nstenergy                = 250\n";
print MDP "nstxtcout                = 25\n";
print MDP "xtc-precision            = 1000\n";
print MDP "nstlist                  = 10\n";
print MDP "pbc                      = xyz\n";
print MDP "rlist                    = 1.1\n";
print MDP "coulombtype              = PME\n";
print MDP "rcoulomb                 = 1.1\n";
print MDP "vdw-type                 = Cut-off\n";
print MDP "rvdw                     = 1.1\n";
print MDP "constraints              = none\n";
close(MDP);
`grompp -f em.mdp -c conf.gro -p topol.top -o em.tpr >& grompp.log`;
print "Energy minimizing...\n";
`mdrun -deffnm em -v >& mdrun.log`;
print "Energy minimization finished\n";
}


sub AddSequence { #reverse yes/no; next chain yes/no; sequence of domain
  my $reverse = shift(@_);
  my $next = shift(@_);
  my @additionsequence = @_;
  
  for (my $p = 0; $p <= $#additionsequence; $p++) {
    my $newchaincheck = 0;
    my $add = $additionsequence[$p];
    #check if new chain will be added in adding this part
    if (($structure[$add] eq "site1" or $structure[$add] eq "site2") and $next) {$newchaincheck++;}
    
    my $structurefile = &ChooseRandomStructure("$structure[$add]");
    print LOG "adding $structurefile to $currfile\n";
    
    if ($reverse) {
      my $ores = $overlap{"$structure[$add+1]-$structure[$add]"};
      MAKENDX($currfile, $ores, "assembly.ndx", $chainsequence[$currchains]);
      `cat ndx.log > assembly-ndx-$chainsequence[$currchains]-$structure[$add].log`;
      MAKENDX($structurefile, $ores, "fit.ndx", "Y");
      `cat ndx.log > fit-ndx-$chainsequence[$currchains]-$structure[$add].log`;
      GCONFRMS($currfile, "assembly.ndx", $structurefile, "fit.ndx");
      `cat fit.log > fit-$chainsequence[$currchains]-$structure[$add].log`;
      `cat fit.pdb > fit-$chainsequence[$currchains]-$structure[$add].pdb`;
      @fit = ();
      @fit = &RemoveOverlappingResiduesFromFit($ores, "Y", "fit-$chainsequence[$currchains]-$structure[$add].pdb");
      &RenameChainsFit("Y", $chainsequence[$currchains]);
      @buffer = ();
      push(@buffer, @{$assembly[$currchains]});
      @{$assembly[$currchains]} = ();
      push(@{$assembly[$currchains]}, @fit);
      push(@{$assembly[$currchains]}, @buffer);	
    }
    else {
      my $ores = $overlap{"$structure[$add-1]-$structure[$add]"};
      MAKENDX($currfile, $ores, "assembly.ndx", $chainsequence[$currchains]);
      `cat ndx.log > assembly-ndx-$chainsequence[$currchains]-$structure[$add].log`;
      MAKENDX($structurefile, $ores, "fit.ndx", "Y");
      `cat ndx.log > fit-ndx-$chainsequence[$currchains]-$structure[$add].log`;
      GCONFRMS($currfile, "assembly.ndx", $structurefile, "fit.ndx");
      `cat fit.log > fit-$chainsequence[$currchains]-$structure[$add].log`;
      `cat fit.pdb > fit-$chainsequence[$currchains]-$structure[$add].pdb`;
      @fit = ();
      @fit = &RemoveOverlappingResiduesFromFit($ores, "Y", "fit-$chainsequence[$currchains]-$structure[$add].pdb");
      &RenameChainsFit("Y", $chainsequence[$currchains]);
      push(@{$assembly[$currchains]}, @fit);
    }
    
    if ($newchaincheck) {
      @fit = ();
      @fit = &AddNewChain("Z", "fit-$chainsequence[$currchains]-$structure[$add].pdb");
      &RenameChainsFit("Z", $chainsequence[$currchains+1]);
      push(@{$assembly[$currchains+1]}, @fit);
      $currfile = &WritePdb($currchains+1, $add);
    }
    
    $currfile = &WritePdb($currchains, $add);
  }
  `rm *#`;
}

sub ChooseRandomStructure {
  my $structure = shift(@_);
  my @structurefiles = split(/\n/, `ls $structure/*pdb`);
  my $choice = $#structurefiles;
  my $rn = $choice*rand();
  my $rounded = int($rn);
  return "$structurefiles[$rounded]";
}

sub AddNewChain {
  my $remchain = shift(@_);
  my $fitfile = shift(@_);
  my @pdblines = ();
  open(FIT, "$fitfile"); 
  while (my $l = <FIT>) {
    if (comment($l) == 1) {
      %d1 = rpdb($l);
      my $chain = $d1{"chain"};
      if ($chain eq $remchain) {
	push(@pdblines, $l);
      }
    }
  }
  close(FILE);
  return @pdblines;  
}


sub RemoveOverlappingResiduesFromFit {
  my $range = shift(@_);
  my @pdblines = ();
  (my $lower, my $upper) = split(/\-/, $range); 
  my $remchain = shift(@_);
  my $fitfile = shift(@_);
  print LOG " removing residues $lower-$upper from $fitfile\n";
  open(FIT, "$fitfile"); 
  while (my $l = <FIT>) {
    if (comment($l) == 1) {
      %d1 = rpdb($l);
      my $chain = $d1{"chain"};
      my $res = $d1{"resnr"};
      if ($chain eq $remchain) {
	if ($res < $lower or $res > $upper) {
	  push(@pdblines, $l);
	}
      }
    }
  }
  close(FIT);
  return @pdblines;  
}


sub MAKENDX {
  my $pdbfile = shift(@_);
  my $range = shift(@_);
  my $ndxfile = shift(@_);
  my $chain = shift(@_);
  #chains Y will be aligned!
  open(NDX, "|make_ndx -f $pdbfile -o $ndxfile >& ndx.log");
  print NDX "keep 3\n";
  print NDX "0 & chain $chain & r $range\n";
  print NDX "keep 1\n";
  print NDX "q\n";
  close(NDX);
}

sub WritePdb {
  my $last = shift(@_);
  my $chain = $chainsequence[$last];
  my $addition = shift(@_);
  my $domain = $structure[$addition];
  my $filename = "assembly-chain$chain-$domain.pdb";
  open (PDB, ">$filename");
  for (my $ch = 0; $ch <= $last; $ch++) {
    for (my $l = 0; $l <= $#{$assembly[$ch]}; $l++) {
      print PDB "$assembly[$ch][$l]";
    }
  }
  close(PDB);
  return $filename;
}

sub WritePdbFinal {
  my $last = shift(@_);
  my $filename = "H-NS-assembly_$last.pdb";
  open (PDB, ">$filename");
  for (my $c = 0; $c < $last; $c++) {
    for (my $l = 0; $l <= $#{$assembly[$c]}; $l++) {
      print PDB "$assembly[$c][$l]";
    }
  }
  close(PDB);
  return $filename;
}

sub AddPdb {
  my $pdbfile = shift(@_);
  my $chain = shift(@_);
  my @pdblines = ();
  open (PDB, "$pdbfile") or die "Could not open file $pdbfile\n";
  while ($l = <PDB>) {
    if (comment($l) == 1) {
      %d1 = &rpdb($l);
      if ($d1{"chain"} eq $chain) {
	push(@pdblines, $l);
      }
    }
  }
  close(FILE);
  return @pdblines;
}

sub RenameChainsAssembly {
  my $last = shift(@_);
  my $currchain = 0;
  for (my $ch = 0; $ch <= $last; $ch++) {
    for (my $l = 0; $l <= $#{$assembly[$ch]}; $l++) {
      %d1 = &rpdb($assembly[$ch][$l]);
      $assembly[$ch][$l] = sprintf("%-6s%5s %-3s%1s%-3s %1s%4s%12s%8s%8s%6s%6s\n", 
				  $d1{"entry"}, 
				  $d1{"atomnr"}, 
				  $d1{"atomname"},
				  $d1{"altloc"},
				  $d1{"resname"},
				  $chainsequence[$currchain],
				  $d1{"resnr"},
				  $d1{"x"},
				  $d1{"y"},
				  $d1{"z"},
				  $d1{"occ"},
				  $d1{"B"});
    }
    $currchain++;
  }
  return $currchain;
}
 

sub RenameChainsFit {
  my $currchain = shift(@_);
  my $newchain = shift(@_);
  for (my $l = 0; $l <= $#fit; $l++) {
    %d1 = rpdb($fit[$l]);
    my $chain = $d1{"chain"};
    if ($chain eq $currchain) {
      $fit[$l] = sprintf("%-6s%5s %-3s%1s%-3s %1s%4s%12s%8s%8s%6s%6s\n", 
			 $d1{"entry"}, 
			 $d1{"atomnr"}, 
			 $d1{"atomname"},
			 $d1{"altloc"},
			 $d1{"resname"},
			 $newchain,
			 $d1{"resnr"},
			 $d1{"x"},
			 $d1{"y"},
			 $d1{"z"},
			 $d1{"occ"},
			 $d1{"B"});
    }
  }
  return $currchain;
}
 


sub GCONFRMS {
  my $f1 = shift(@_);
  my $n1 = shift(@_);
  my $f2 = shift(@_);
  my $n2 = shift(@_);
  my $fit = "fit.pdb";
  `g_confrms -f1 $f1 -f2 $f2 -n1 $n1 -n2 $n2 -o $fit -one >& fit.log`;
}

sub comment {
    my $a = shift(@_);
    if ($a =~ /^ATOM/ or $a =~ /^HETATM/) { return 1; }
    else { return 0;}  
}

sub rpdb {
  my $l = shift(@_);
  my %a;
  $a{"entry"}    = substr($l, 0, 6);
  $a{"atomnr"}   = substr($l, 6, 5);
  $a{"atomname"} = substr($l, 12, 4);
  $a{"altloc"}   = substr($l, 16, 1);
  $a{"resname"}  = substr($l, 17, 3);
  $a{"resnr"}    = substr($l, 22, 4);
  $a{"chain"}    = substr($l, 21, 1);
  $a{"x"}        = substr($l, 30, 8);
  $a{"y"}        = substr($l, 38, 8);
  $a{"z"}        = substr($l, 46, 8);
  $a{"occ"}      = substr($l, 54, 6);
  $a{"B"}        = substr($l, 60, 6);
  return %a;
}

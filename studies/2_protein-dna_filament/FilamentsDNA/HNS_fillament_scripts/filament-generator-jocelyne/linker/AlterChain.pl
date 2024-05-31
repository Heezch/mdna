#!/usr/bin/perl

unless(@ARGV) { die "usage: pdb.pl file\n";}
my $f = shift(@ARGV);

my $chain = "Y"; 
my $resnr = 0;

open (FILE, "$f") or die "Could not open file $name\n";
while ($l = <FILE>) {
    if (comment($l) == 1) {
	%d1 = rpdb($l);
	if ($resnr == 0) {$resnr = $d1{"resnr"};}
	elsif ($d1{"resnr"} < $resnr or $d1{"resnr"} > $resnr+1) {$chain = "Z";}
	else {$resnr = $d1{"resnr"};}
	if ($d1{"chain"} =~ /\d/) {$chain = $d1{"chain"};}
	printf("%-6s%5s %-3s%1s%-3s %1s%4s%12s%8s%8s%6s%6s\n", 
	       $d1{"entry"}, 
	       $d1{"atomnr"}, 
	       $d1{"atomname"},
	       $d1{"altloc"},
	       $d1{"resname"},
	       "$chain",
	       $d1{"resnr"},
	       $d1{"x"},
	       $d1{"y"},
	       $d1{"z"},
	       $d1{"occ"},
	       $d1{"B"});
    }
}
close(FILE);

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
sub rxyz {
    my $l = shift(@_);
    my %a;
    $a{"x"} = substr($l, 31, 8);
    $a{"y"} = substr($l, 39, 8);
    $a{"z"} = substr($l, 47, 8);
    return %a;
}

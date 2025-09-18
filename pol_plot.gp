reset
set terminal pngcairo size 1200,800 enhanced font "DejaVuSans,12"

do for [i=1:118] {
    set output sprintf("POL/png/pol_%d.png", i)
	plot "POL/pol".i.".out" using 1:2 w l title "P1","" using 1:3 w l title "P2", ""  using 1:4 w l title "P3"
    unset output
}


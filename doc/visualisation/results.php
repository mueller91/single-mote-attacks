<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
    <link href="http://cdn.rawgit.com/noelboss/featherlight/1.7.12/release/featherlight.min.css" type="text/css" rel="stylesheet" />
    <script src="http://code.jquery.com/jquery-latest.js"></script>
    <script src="http://cdn.rawgit.com/noelboss/featherlight/1.7.12/release/featherlight.min.js" type="text/javascript" charset="utf-8"></script>
    <title>RPL Attack Results</title>
</head>
<body>
<!-- HOW TO USE -->
<!-- Latest compiled and minified CSS -->
<div class="jumbotron">
    <div class="container">

        <ul class="nav nav-pills" style="position: fixed; top: 5em; background-color:white; border: 1px solid black;">
            <li class="active"><a data-toggle="pill" href="#blackhole-attack">Blackhole Attack</a></li>
            <li><a data-toggle="pill" href="#hello-flood-attack">Hello Flood Attack</a></li>
            <li><a data-toggle="pill" href="#version-number-attack">Version Number Attack</a></li>
        </ul>
        <input type='button' id="switch" value='Switch to C' onclick="switch_to_c();" style='position: fixed; top: 5em; right: 20em;'/>
        <?php

        $datasets = array(
            "1l",
            "2l",
            "3l",
            "4l",
            "5l",
            "6l",
            "7l",
            "8l",
            "9l",
            "10l",
            "11l",
            "12l",
            "13l",
            "14l",
            "15l",
            "16l",
            "17l",
            "18l",
            "19l",
            "20l"
        );
        $button_position = 5;
        for($i=0; $i<=count($datasets)-1; $i++) {
            $button_coord = strval($button_position).'em';
            echo "<input type='button' value='$datasets[$i]' onclick=\"switch_to_dataset('$datasets[$i]');\" style='position: fixed; top: $button_coord; right: 5em;'/>\n\t";
            $button_position += 3;
        }
        ?>

        <script>
            function switch_to_dataset(dataset) {
                let path;
                let current_dataset = document.getElementById("dataset").innerText;

                let images = document.getElementsByTagName("img");
                for (let i = 0; i < images.length; i++) {
                    path = images[i].src;
                    images[i].src = path.replace('/' + current_dataset + '/', '/' + dataset + '/');
                }

                let anchors = document.getElementsByTagName("a");
                for (let j = 0; j < anchors.length; j++) {
                    path = anchors[j].getAttribute('data-featherlight');
                    if (path != null) {
                        anchors[j].setAttribute('data-featherlight', path.replace('/' + current_dataset + '/', '/' + dataset + '/'));
                    }
                }

                document.getElementById("dataset").innerText = dataset;
            }

        </script>
        <script>

            function switch_to_c() {
                let path;
                let switchedToC;

                let button = document.getElementById("switch");
                if (button.value === 'Switch to C') {
                    button.value = 'Switch to Python';
                    switchedToC = true;
                } else {
                    button.value = 'Switch to C';
                    switchedToC = false;
                }

                // switch images!
                let images = document.getElementsByTagName("img");
                for (let i = 0; i < images.length; i++) {
                    path = images[i].src;
//                    // AREA IMAGES
//                    if (switchedToC === true && path.includes('py_area.png')) {
//                        images[i].src = path.replace('py_area.png', 'c_area.png');
//                    }
//                    else if (switchedToC === false && path.includes('c_area.png')) {
//                        images[i].src = path.replace('c_area.png', 'py_area.png');
//                    }
                    // CLASSIFICATION IMAGES
                    if (switchedToC === true && path.includes('py_classification.png')) {
                        images[i].src = path.replace('py_classification.png', 'c_classification.png');
                    }
                    else if (switchedToC === false && path.includes('c_classification.png')) {
                        images[i].src = path.replace('c_classification.png', 'py_classification.png');
                    }
                }

                // switch anchors!
                let anchors = document.getElementsByTagName("a");
                for (let j = 0; j < anchors.length; j++) {
                    path = anchors[j].getAttribute('data-featherlight');
                    if (path == null) {
                        continue;
                    }
//                    AREA PLOT IS ALWAYY THE SAME
//                    // AREA
//                    if (switchedToC === true != null && path.includes('py_area.png')) {
//                        images[i].src = path.replace('py_area.png', 'c_area.png');
//                        path = path.replace('py_area.png', 'c_area.png');
//                        anchors[j].setAttribute('data-featherlight', path);
//                    }
//                    else if (switchedToC === false && path.includes('individual/c/')) {
//                        images[i].src = path.replace('c_area.png', 'py_area.png');
//                        path = path.replace('c_area.png', 'py_area.png');
//                        anchors[j].setAttribute('data-featherlight', path);
//                    }
                    // CLASSIFICATION
                     if (switchedToC === true != null && path.includes('py_classification.png')) {
                        path = path.replace('py_classification.png', 'c_classification.png');
                        anchors[j].setAttribute('data-featherlight', path);
                    }
                    else if (switchedToC === false && path.includes('c_classification.png')) {
                        path = path.replace('c_classification.png', 'py_classification.png');
                        anchors[j].setAttribute('data-featherlight', path);

                    }
                }
            }
        </script>

        <div class="tab-content">
            <table>
                <tr>
                </tr>
            </table>
            <?php
            $nodes = array(
                "c1:0c:00:00:00:00:00:00",
                "c1:0c:00:00:00:00:00:01",
                "c1:0c:00:00:00:00:00:02",
                "c1:0c:00:00:00:00:00:03",
                "c1:0c:00:00:00:00:00:04",
                "c1:0c:00:00:00:00:00:05",
                "c1:0c:00:00:00:00:00:06",
                "c1:0c:00:00:00:00:00:07",
                "c1:0c:00:00:00:00:00:08",
                "c1:0c:00:00:00:00:00:09",
                "c1:0c:00:00:00:00:00:0a",
                "c1:0c:00:00:00:00:00:0b",
                "c1:0c:00:00:00:00:00:0c",
                "c1:0c:00:00:00:00:00:0d",
                "c1:0c:00:00:00:00:00:0e",
                "c1:0c:00:00:00:00:00:0f"
            );

            $nodes_short = array(
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"
            );

            $tests = array(
                "blackhole-attack",
                "hello-flood-attack",
                "version-number-attack",
            );

            $rand_query_string = rand(0, 999999999);
            for ($i = 0; $i <= count($tests) - 1; $i++) {
                /** CREATE TABS */
                if ($i < 1)
                    echo "<div id='$tests[$i]' class='tab-pane fade in active'>\n";
                if ($i > 0)
                    echo "<div id='$tests[$i]' class='tab-pane fade'>\n";
                echo "<h3> $tests[$i] </h3>\n";
                echo "<table>\n<thead>\n";
                echo "<strong style='position: fixed; top: 1em; right: 5em;'>Current Data Set:</strong>\n";
                echo "<strong id='dataset' style='position: fixed; top: 3em; right: 5em;'>1l</strong>\n";

                /* DODAGS */
                $dodags            = array(
                    "~/../../../../RPL-Attacks-Data/1l/control-set-clean/without-malicious/results/udp_flow_graph.png?no_cache=$rand_query_string",
                    "~/../../../../RPL-Attacks-Data/1l/$tests[$i]/without-malicious/results/udp_flow_graph.png?no_cache=$rand_query_string",
                    "~/../../../../RPL-Attacks-Data/1l/$tests[$i]/with-malicious/results/udp_flow_graph.png?no_cache=$rand_query_string",
                );
                $dodag_position = 5;
                for ($j = 0; $j <= count($dodags) - 1; $j++) {
                    echo "<a href='#' data-featherlight='$dodags[$j]'>  \n";
                    echo "<img height='300' src='$dodags[$j]' style='position: fixed; top: $dodag_position";
                    echo "em; left: 0; border: 3px solid grey; width: 340px'>\n";
                    echo "</a>";
                    echo "\n";
                    $dodag_position += 25;
                }

                /* RESULTS */
                for ($x = 0; $x <= count($nodes) - 1; $x++) {
                    $path_train_area     = "~/../../../../RPL-Attacks-Data/1l/control-set-clean/without-malicious/data/individual/$nodes_short[$x]_area.png?no_cache=$rand_query_string";
                    $path_test_good_area = "~/../../../../RPL-Attacks-Data/1l/$tests[$i]/without-malicious/data/individual/$nodes_short[$x]_area.png?no_cache=$rand_query_string";
                    $path_test_mal_area  = "~/../../../../RPL-Attacks-Data/1l/$tests[$i]/with-malicious/data/individual/$nodes_short[$x]_area.png?no_cache=$rand_query_string";
                    /* TRAIN DATA - WITHOUT MALICIOUS */
                    echo "<tr>\n";
                    echo "       <td><h4>Control set: $nodes[$x]</h4>";
                    echo "       <a href='#' data-featherlight='$path_train_area'>  \n";
                    echo "       <img height='300' src='$path_train_area' class='img-fluid' style='border: 3px solid grey'> </a>\n";
                    echo "       </td>\n";

                    /* TEST - WITHOUT MALICIOUS */
                    echo "       <td><h4>Training set: $nodes[$x]</h4>";
                    echo "       <a href='#' data-featherlight='$path_test_good_area'>  \n";
                    echo "       <img height='300' src='$path_test_good_area' class='img-fluid' style='border: 3px solid grey' hspace='10'> </a>\n";
                    echo "       </td>\n";

                    /* TEST - WITH MALICIOUS */
                    echo "       <td><h4>Test set: $nodes[$x]</h4>";
                    echo "       <a href='#' data-featherlight='$path_test_mal_area'>  \n";
                    echo "       <img height='300' src='$path_test_mal_area' class='img-fluid' style='border: 3px solid grey'> </a>\n";
                    echo "       </td>\n";
                    echo "</tr>\n";

                    echo "<tr>\n";
                    /* CLASSIFICATION */
                    $path_class_control = "~/../../../../RPL-Attacks-Data/1l/control-set-clean/without-malicious/data/individual/$nodes_short[$x]_py_classification.png?no_cache=$rand_query_string";
                    $path_class_good = "~/../../../../RPL-Attacks-Data/1l/$tests[$i]/without-malicious/data/individual/$nodes_short[$x]_py_classification.png?no_cache=$rand_query_string";
                    $path_class_mal  = "~/../../../../RPL-Attacks-Data/1l/$tests[$i]/with-malicious/data/individual/$nodes_short[$x]_py_classification.png?no_cache=$rand_query_string";
                    echo "      <td><a href='#' data-featherlight='$path_class_control'>  \n";
                    echo "          <img height='153' src=$path_class_control class='img-fluid' style='border: 3px solid grey'>\n";
                    echo "      </a></td>\n";
//        echo "<td></td>";
                    echo "\n";
                    echo "      <td><a href='#' data-featherlight='$path_class_good'>  \n";
                    echo "          <img height='153' src=$path_class_good class='img-fluid' style='border: 3px solid green' hspace='10'>\n";
                    echo "      </a></td>\n";
                    echo "\n";
                    echo "      <td><a href='#' data-featherlight='$path_class_mal'>  \n";
                    echo "          <img height='153' src=$path_class_mal class='img-fluid' style='border: 3px solid red'>\n";
                    echo "      </a></td>\n";
                    echo "</tr>\n";
                }
                echo "</thead>\n</table>\n\n";
                echo "</div>\n\n";
            }
            ?>
        </div>
    </div>
</div>
</body>
</html>
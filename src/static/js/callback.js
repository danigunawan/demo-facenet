// var API = "/detectobject/";
var API = "/";
var canvas = document.getElementById('myChart');
var data = {
  labels: ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
  datasets: [{
    label: "My First dataset",
    fill: false,
    lineTension: 0.2,
    backgroundColor: "rgba(75,192,192,0.4)",
    borderColor: "rgba(75,192,192,1)",
    borderCapStyle: 'butt',
    borderDash: [],
    borderDashOffset: 0.0,
    borderWidth: 1,
    borderJoinStyle: 'miter',
    pointBorderColor: "rgba(75,192,192,1)",
    pointBackgroundColor: "#fff",
    pointBorderWidth: 1,
    pointHoverRadius: 5,
    pointHoverBackgroundColor: "rgba(75,192,192,1)",
    pointHoverBorderColor: "rgba(220,220,220,1)",
    pointHoverBorderWidth: 2,
    pointRadius: 0,
    pointHitRadius: 10,
    data: [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null],
  }]
};

$(document).ready(function(){
    $("#blackListList").children().remove();

    $.ajax({
        type: "POST",
        url: API + "getblack_list",
        processData: false,
        contentType: false,
        success: function (data) {
            console.log(data);
            var strVar="";
            for(i = 0; i < data["face_black_list"].length; i++){
                strVar += "<div class=\"w3-container\">";
                strVar += "                            <div class=\"w3-card-4\" style=\"width:100%;padding: 5px;margin-top: 10px;\">";
                strVar += "                                <div class=\"col-md-6\" style=\"padding: 0px 5px 0px 0px;margin-bottom: 10px;\">";
                strVar += "                                    <img src=\""+ data["face_black_list"][i]["image"]["path_image"] +"\" class=\"img-rounded\" width=\"220\" height=\"140\">";
                strVar += "                                <\/div>";
                strVar += "                                <div class=\"col-md-6\" style=\"padding: 0px 0px 0px 5px;\">";
                strVar += "                                    <img src=\""+ data["face_black_list"][i]["image_cmt"]["path_image"] +"\" class=\"img-rounded\" width=\"220\" height=\"140\">";
                strVar += "                                <\/div>";
                strVar += "                                <div class=\"w3-container\">";
                strVar += "                                    <p>Tên: "+ data["face_black_list"][i]["name"] +"<\/p>";
                strVar += "                                    <p>Số CMT: "+ data["face_black_list"][i]["id"] +"<\/p>";
                strVar += "                                <\/div>";
                strVar += "                            <\/div>";
                strVar += "                        <\/div>";
            }

            $("#blackListList").append(strVar);
        }
    });
});

function adddata(score) {
    myLineChart.data.datasets[0].data.shift();
    myLineChart.data.labels.shift();
    myLineChart.data.datasets[0].data.push(score);
    var ctime = new Date();
    var csecs = moment(ctime).format("s");
    
    var label = csecs == '0' ? moment(ctime).format("mm:ss") : moment(ctime).format(":ss");
    myLineChart.data.labels.push(label);
    
    myLineChart.update();

}

var option = {
    showLines: true,
    animation: false,
    legend: {
        display: false
    },
    scales: {
        yAxes: [{
        ticks: {
            max: 100,
            min: 0,
            stepSize: 50
        },
        gridLines: {
            drawTicks: false
        }
    }],
    xAxes: [{
        gridLines: {
            display: true,
            drawTicks: false
        },
        ticks: {
            fontSize: 10,
            maxRotation: 10,
            callback: function(value) {
                if (value.toString().length > 0) {
                    return value;
                } else {return null};
            }
        }
    }]
    }
};
var myLineChart = Chart.Line(canvas, {
    data: data,
    options: option
});

/** Images scanned so far. */
var imagesScanned = [];

// -------------- Optional status display, depending on JQuery --------------
function displayStatus(loading, mesg, clear) {
    if(loading) {
        $('#info').html((clear ? '' : $('#info').html()) + '<p><img src="https://asprise.com/legacy/product/_jia/applet/loading.gif" style="vertical-align: middle;" hspace="8"> ' + mesg + '</p>');
    } else {
        $('#info').html((clear ? '' : $('#info').html()) + mesg);
    }
}

function scanToJpg() {
    $(".progress_active").removeClass('progress_active');
    $("#u1413").addClass('progress_active');

    $("#scanned_cmt").hide();
    $("#loader_img_1").show();
    scanner.scan(displayImagesOnPage,
        {
            "output_settings": [
                {
                    "type": "return-base64",
                    "format": "jpg"
                }
            ]
        }
    );
}

/** Returns true if it is successfully or false if failed and reports error. */
function checkIfSuccessfully(successful, mesg, response) {
    displayStatus(false, '', true);
    if(successful && mesg != null && mesg.toLowerCase().indexOf('user cancel') >= 0) { // User cancelled.
        displayStatus(false, '<pre>' + "User cancelled." + '</pre>', true);
        return false;
    } else if(!successful) {
        displayStatus(false, '<pre>' + "Failed: " + mesg + "\n" + response + '</pre>', true);
        return false;
    }
    return true;
}

/** Processes the scan result */
function displayImagesOnPage(successful, mesg, response) {
    if(!successful) { // On error
        console.error('Failed: ' + mesg);
        return;
    }

    if(successful && mesg != null && mesg.toLowerCase().indexOf('user cancel') >= 0) { // User cancelled.
        console.info('User cancelled');
        return;
    }

    var scannedImages = scanner.getScannedImages(response, true, false); // returns an array of ScannedImage
    for(var i = 0; (scannedImages instanceof Array) && i < scannedImages.length; i++) {
        var scannedImage = scannedImages[i];
        processScannedImage(scannedImage);
    }
}

/** Processes a ScannedImage */
function processScannedImage(scannedImage) {
    imagesScanned.push(scannedImage);
    $("#loader_img_1").hide();
    $("#scanned_cmt").show();
    document.getElementById("scanned_cmt").src = scannedImage.src;
    savedata_callback();
}

function savedata_callback(){
    $(".progress_active").removeClass('progress_active');
    $("#u1406").addClass('progress_active');

    urltoFile(imagesScanned[0]["src"], 'cmt.jpg', 'image/jpg')
    .then(function(file){
        formData = new FormData();
        cmt_id = $("#chungminhthu").val();
        formData.append("id", cmt_id);
        formData.append("myfile", file);

        $("#loader_img_2").show();

        $.ajax({
            type: "POST",
            url: API + "insertDataRaw",
            data: formData,
            processData: false,
            contentType: false,
            success: function (data) {
                $("#loader_img_2").hide();
                if(data["status"] == "success"){
                    document.getElementById("digital_img").src = data["image_lastest"].replace("/data/image-processing/facenet/data", "/facenet");
                    $("#digital_img").show(); 
                    $("#finishTransaction").removeAttr("disabled");
                    blacklist_callback();
                }else{
                    $("#process").hide();
                    a = confirm("Không nhận được mặt, chụp lại?");
                    if(a){
                        savedata_callback();
                    }
                }
                
            }
        });
    })
}

//callback
function blacklist_callback() {
    $(".progress_active").removeClass('progress_active');
    $("#u1488").addClass('progress_active');

    $("#black_list_result").hide();
    $("#exist_user_result").hide();
    $("#new_user_result").hide();

    formData = new FormData();
    cmt_id = $("#chungminhthu").val();
    formData.append("id", cmt_id);

    $.ajax({
        type: "POST",
        url: API + "check_black_list",
        data: formData,
        processData: false,
        contentType: false,
        success: function (data) {
            $("#myModal .modal-body").children().remove();
            
            if(data["status"] == 1){
                //có trong black list
                document.getElementById("u4488_img").src = "static/images/alert-icon-1562.png?crc=523315291";
                $("#u4993-6 p").text('CLEAN: Người này nằm trong danh sách đen');
                $("#u3864-4 p").text('NẰM TRONG SANH SÁCH ĐEN');
                $("#label_result").attr("data-target", "#myModal");
                $("#label_result").show();

                $("#u3856-4 p").text('NGƯỜI NÀY NẰM TRONG DANH SÁCH ĐEN');

                var strVar="";
                for (i = 0; i < data["result"].length; i++) {
                    var strVar="";
                    strVar += "<div class=\"position_content\" id=\"u3853_position_content\">";
                    strVar += "                        <div class=\"clearfix colelem\" id=\"ppu3858-4\"><!-- group -->";
                    strVar += "                            <div class=\"clearfix grpelem\" id=\"pu3858-4\"><!-- column -->";
                    strVar += "                                <div class=\"clearfix colelem\" id=\"u3858-4\"><!-- content -->";
                    strVar += "                                    <p>NHẬN DẠNG KHUÔN MẶT<\/p>";
                    strVar += "                                <\/div>";
                    strVar += "                                <div class=\"clearfix colelem\" id=\"pu3879\"><!-- group -->";
                    strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3879\"><!-- group -->";
                    strVar += "                                        <div class=\"clearfix grpelem\" id=\"u3892-4\"><!-- content -->";
                    strVar += "                                            <img class=\"img-rounded\" width=\"250\" height=\"160\" style=\"position: absolute;\" src='"+ data["result"][i]["image_query"].replace("/data/image-processing/facenet/data", "/facenet") +"'>";
                    strVar += "                                        <\/div>";
                    strVar += "                                    <\/div>";
                    strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3883\"><!-- group -->";
                    strVar += "                                        <div class=\"clearfix grpelem\" id=\"u3905-4\"><!-- content -->";
                    strVar += "                                            <img class=\"img-rounded\" width=\"250\" height=\"160\" style=\"position: absolute;\" src='"+ data["result"][i]["image_black_list"].replace("/data/image-processing/facenet/data", "/facenet") +"'>";
                    strVar += "                                        <\/div>";
                    strVar += "                                    <\/div>";
                    strVar += "                                <\/div>";
                    strVar += "                                <div class=\"clearfix colelem\" id=\"u3888\"><!-- group -->";
                    strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3891-4\"><!-- content -->";
                    strVar += "                                        <p>Giống nhau "+ parseFloat(parseFloat(data["result"][i]["score"])*100).toFixed(2) +"%<\/p>";
                    strVar += "                                    <\/div>";
                    strVar += "                                <\/div>";
                    strVar += "                            <\/div>";
                    strVar += "                            <div class=\"clearfix grpelem\" id=\"u3859-4\"><!-- content -->";
                    strVar += "                                <p>HỒ SƠ<\/p>";
                    strVar += "                            <\/div>";
                    strVar += "                            <div class=\"grpelem\" id=\"u3899\"><!-- simple frame --><\/div>";
                    strVar += "                            <div class=\"grpelem\" id=\"u3900\"><!-- simple frame --><\/div>";
                    strVar += "                            <div class=\"rgba-background clearfix grpelem\" id=\"u3901\"><!-- column -->";
                    strVar += "                                <div class=\"clearfix colelem\" id=\"pu3903\"><!-- group -->";
                    strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3903\"><!-- group -->";
                    strVar += "                                        <div class=\"clearfix grpelem\" id=\"u3906-4\"><!-- content -->";
                    strVar += "                                            <img class=\"img-rounded\" width=\"250\" height=\"160\" style=\"position: absolute;\" src='"+ data["info"][i]["image"]["path_image"].replace("/data/image-processing/facenet/data", "/facenet") +"'>";
                    strVar += "                                        <\/div>";
                    strVar += "                                    <\/div>";
                    strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3904\"><!-- group -->";
                    strVar += "                                        <div class=\"clearfix grpelem\" id=\"u3907-4\"><!-- content -->";
                    strVar += "                                            <img class=\"img-rounded\" width=\"250\" height=\"160\" style=\"position: absolute;\" src='"+ data["info"][i]["image_cmt"]["path_image"].replace("/data/image-processing/facenet/data", "/facenet") +"'>";
                    strVar += "                                        <\/div>";
                    strVar += "                                    <\/div>";
                    strVar += "                                <\/div>";
                    strVar += "                                <div class=\"clearfix colelem\" id=\"u3902-35\"><!-- content -->";
                    strVar += "                                    <p>NAME: "+ data["info"][i]["name"] +"<\/p>";
                    strVar += "                                    <p>&nbsp;<\/p>";
                    strVar += "                                    <p>Gender:<\/p>";
                    strVar += "                                    <p>&nbsp;<\/p>";
                    strVar += "                                    <p>Date of birth:<\/p>";
                    strVar += "                                    <p>&nbsp;<\/p>";
                    strVar += "                                    <p>Home Adress:<\/p>";
                    strVar += "                                    <p>&nbsp;<\/p>";
                    strVar += "                                    <p>Email:<\/p>";
                    strVar += "                                    <p>&nbsp;<\/p>";
                    strVar += "                                    <p>Phone Number:<\/p>";
                    strVar += "                                <\/div>";
                    strVar += "                            <\/div>";
                    strVar += "                        <\/div>";
                    strVar += "                    <\/div>";

                    break;
                }
                $("#myModal .modal-body").append(strVar);
                $("#blacklistKeepTracking").attr("onclick", "interval_check_realtime('"+ data["info"][0]["id"] +"',0,0);");
                $("#myModal").modal("show");
            }
            else if (data["status"] == 2){
                $("#process").hide();
                alert("Không nhận diện được mặt. Chụp lại");
            }
            else{
                //không có trong black list gọi api tiếp theo
                check_similarity_callback();
            }
        }
    });
}

function check_similarity_callback(){
    $(".progress_active").removeClass('progress_active');
    $("#u1497").addClass('progress_active');

    $("#newUserModal .modal-body").children().remove();
    $("#myModal .modal-body").children().remove();   
    
    //Usage example:
    formData = new FormData();
    cmt_id = $("#chungminhthu").val();
    formData.append("id_cmt", cmt_id);

    $.ajax({
        type: "POST",
        url: API + "checkSimilarity",
        data: formData,
        processData: false,
        contentType: false,
        success: function (data) {
            $("#process").hide();

            if(data["status"] == 0){
                //người mới trong CSDL
                $("#u3864-4 p").text('NGƯỜI MỚI');
                
                if(parseFloat(parseFloat(data["detail_res_compare"][0]["score"])*100).toFixed(2) < 90){
                    $("#newUserModal .modal-header").css('background-color', '#FF9898');
                    document.getElementById("u4488_img").src = "static/images/alert-icon-1562.png?crc=523315291";
                    $("#u4993-6 p").text("WARNING: Người này chỉ giống "+ parseFloat(parseFloat(data["detail_res_compare"][0]["score"])*100).toFixed(2) +"% so với CMT");
                }else{
                    $("#newUserModal .modal-header").css('background-color', '#82B2CE');
                    document.getElementById("u4488_img").src = "static/images/check-mark--crop-u5972.png?crc=3822812632";
                    $("#u4993-6 p").text("CLEAN: Người này giống "+ parseFloat(parseFloat(data["detail_res_compare"][0]["score"])*100).toFixed(2) +"% so với CMT");
                }
                $("#label_result").attr("data-target", "#newUserModal");
                $("#label_result").show();

                var strVar = "";
                strVar = "";
                for (i = 0; i < data["detail_res_compare"].length; i++) {
                    strVar += "<div class=\"row\" style=\"margin-bottom: 10px;\">";
                    strVar += "                            <div class=\"col-md-6 col-lg-6\" align=\"center\" style=\"padding: 0px;\"> ";
                    strVar += "                                <label for=\"exampleInputEmail1\">Ảnh chứng minh thư<\/label>";
                    strVar += "                                <img alt=\"User Pic\" src=\""+ data["path_new_cmt"].replace("/data/image-processing/facenet/data", "/facenet") +"\" class=\"img-rounded\" width=\"250\" height=\"160\">";
                    strVar += "                            <\/div>";
                    strVar += "                            <div class=\"col-md-6 col-lg-6\" align=\"center\" style=\"padding: 0px;\"> ";
                    strVar += "                                <label for=\"exampleInputEmail1\">Ảnh chụp<\/label>";
                    strVar += "                                <img alt=\"User Pic\" src=\""+ data["detail_res_compare"][i]["path_face_compare"].replace("/data/image-processing/facenet/data", "/facenet") +"\" class=\"img-rounded\" width=\"250\" height=\"160\"> ";
                    strVar += "                            <\/div>";
                    strVar += "                            <div class=\"col-md-12 col-lg-12\" align=\"center\"> ";
                    strVar += "                                <span class=\"label label-success\" style=\"display: block;font-size: 16px;\">Giống nhau: "+ parseFloat(parseFloat(data["detail_res_compare"][i]["score"])*100).toFixed(2) +"%<\/span>";
                    strVar += "                            <\/div>";
                    strVar += "                        <\/div>";
                }
                
                $("#new_user_best_score").text("Giống nhau: " + parseFloat(parseFloat(data["detail_res_compare"][0]["score"])*100).toFixed(2) + "%");
                $("#newUserModal .modal-body").append(strVar);
                $("#newUserModalKeepTracking").attr("onclick", "interval_check_realtime('"+ cmt_id +"',1,1);");
                $("#newUserModal").modal("show");
            }
            else if(data["status"] == 1){
                //người đã tồn tại trong CSDL
                $("#u3856-4 p").text('ĐÃ TỪNG GIAO DỊCH TRƯỚC ĐÂY');
                $("#u3864-4 p").text('ĐÃ GIAO DỊCH TRƯỚC ĐÂY');
                
                if(parseFloat(parseFloat(data["face"]["score"]) * 100).toFixed(2) < 90){
                    $("#myModal .modal-header").css('background-color', '#FF9898');
                    document.getElementById("u4488_img").src = "static/images/alert-icon-1562.png?crc=523315291";
                    $("#u4993-6 p").text("WARNING: Người này chỉ giống "+ parseFloat(parseFloat(data["face"]["score"]) * 100).toFixed(2) +"% với những lần giao dịch trước");
                }else{
                    $("#myModal .modal-header").css('background-color', '#82B2CE');
                    document.getElementById("u4488_img").src = "static/images/check-mark--crop-u5972.png?crc=3822812632";
                    $("#u4993-6 p").text("CLEAN: Người này giống "+ parseFloat(parseFloat(data["face"]["score"]) * 100).toFixed(2) +"% với những lần giao dịch trước");
                }
                $("#label_result").attr("data-target", "#myModal");
                $("#label_result").show();    

                var strVar="";
                strVar += "<div class=\"position_content\" id=\"u3853_position_content\">";
                strVar += "                        <div class=\"clearfix colelem\" id=\"ppu3858-4\"><!-- group -->";
                strVar += "                            <div class=\"clearfix grpelem\" id=\"pu3858-4\"><!-- column -->";
                strVar += "                                <div class=\"clearfix colelem\" id=\"u3858-4\"><!-- content -->";
                strVar += "                                    <p>NHẬN DẠNG KHUÔN MẶT<\/p>";
                strVar += "                                <\/div>";
                strVar += "                                <div class=\"clearfix colelem\" id=\"pu3879\"><!-- group -->";
                strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3879\"><!-- group -->";
                strVar += "                                        <div class=\"clearfix grpelem\" id=\"u3892-4\"><!-- content -->";
                strVar += "                                            <img class=\"img-rounded\" width=\"250\" height=\"160\" style=\"position: absolute;\" src='"+ data["cmt"]["path_best_similar"].replace("/data/image-processing/facenet/data", "/facenet") +"'>";
                strVar += "                                        <\/div>";
                strVar += "                                    <\/div>";
                strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3883\"><!-- group -->";
                strVar += "                                        <div class=\"clearfix grpelem\" id=\"u3905-4\"><!-- content -->";
                strVar += "                                            <img class=\"img-rounded\" width=\"250\" height=\"160\" style=\"position: absolute;\" src='"+ data["cmt"]["path_new_image"].replace("/data/image-processing/facenet/data", "/facenet") +"'>";
                strVar += "                                        <\/div>";
                strVar += "                                    <\/div>";
                strVar += "                                <\/div>";
                strVar += "                                <div class=\"clearfix colelem\" id=\"u3888\"><!-- group -->";
                strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3891-4\"><!-- content -->";
                strVar += "                                        <p>Giống nhau "+ parseFloat(parseFloat(data["cmt"]["score"])*100).toFixed(2) +"%<\/p>";
                strVar += "                                    <\/div>";
                strVar += "                                <\/div>";
                strVar += "                                <div class=\"clearfix colelem\" id=\"pu3881\"><!-- group -->";
                strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3881\"><!-- group -->";
                strVar += "                                        <div class=\"clearfix grpelem\" id=\"u3893-4\"><!-- content -->";
                strVar += "                                            <img class=\"img-rounded\" width=\"250\" height=\"160\" style=\"position: absolute;\" src='"+ data["face"]["path_best_similar"].replace("/data/image-processing/facenet/data", "/facenet") +"'>";
                strVar += "                                        <\/div>";
                strVar += "                                    <\/div>";
                strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3884\"><!-- group -->";
                strVar += "                                        <div class=\"clearfix grpelem\" id=\"u3895-4\"><!-- content -->";
                strVar += "                                            <img class=\"img-rounded\" width=\"250\" height=\"160\" style=\"position: absolute;\" src='"+ data["face"]["path_new_image"].replace("/data/image-processing/facenet/data", "/facenet") +"'>";
                strVar += "                                        <\/div>";
                strVar += "                                    <\/div>";
                strVar += "                                <\/div>";
                strVar += "                                <div class=\"clearfix colelem\" id=\"u3889\"><!-- group -->";
                strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3897-4\"><!-- content -->";
                strVar += "                                        <p>Giống nhau "+ parseFloat(parseFloat(data["face"]["score"])*100).toFixed(2) +"%<\/p>";
                strVar += "                                    <\/div>";
                strVar += "                                <\/div>";
                strVar += "                            <\/div>";
                strVar += "                            <div class=\"clearfix grpelem\" id=\"u3859-4\"><!-- content -->";
                strVar += "                                <p>HỒ SƠ<\/p>";
                strVar += "                            <\/div>";
                strVar += "                            <div class=\"grpelem\" id=\"u3899\"><!-- simple frame --><\/div>";
                strVar += "                            <div class=\"grpelem\" id=\"u3900\"><!-- simple frame --><\/div>";
                strVar += "                            <div class=\"rgba-background clearfix grpelem\" id=\"u3901\"><!-- column -->";
                strVar += "                                <div class=\"clearfix colelem\" id=\"pu3903\"><!-- group -->";
                strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3903\"><!-- group -->";
                strVar += "                                        <div class=\"clearfix grpelem\" id=\"u3906-4\"><!-- content -->";
                strVar += "                                            <img class=\"img-rounded\" width=\"250\" height=\"160\" style=\"position: absolute;\" src='"+ data["cmt"]["path_best_similar"].replace("/data/image-processing/facenet/data", "/facenet") +"'>";
                strVar += "                                        <\/div>";
                strVar += "                                    <\/div>";
                strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3904\"><!-- group -->";
                strVar += "                                        <div class=\"clearfix grpelem\" id=\"u3907-4\"><!-- content -->";
                strVar += "                                            <img class=\"img-rounded\" width=\"250\" height=\"160\" style=\"position: absolute;\" src='"+ data["face"]["path_best_similar"].replace("/data/image-processing/facenet/data", "/facenet") +"'>";
                strVar += "                                        <\/div>";
                strVar += "                                    <\/div>";
                strVar += "                                <\/div>";
                strVar += "                                <div class=\"clearfix colelem\" id=\"u3902-35\"><!-- content -->";
                strVar += "                                    <p>NAME: "+ cmt_id +"<\/p>";
                strVar += "                                    <p>&nbsp;<\/p>";
                strVar += "                                    <p>Gender:<\/p>";
                strVar += "                                    <p>&nbsp;<\/p>";
                strVar += "                                    <p>Date of birth:<\/p>";
                strVar += "                                    <p>&nbsp;<\/p>";
                strVar += "                                    <p>Home Adress:<\/p>";
                strVar += "                                    <p>&nbsp;<\/p>";
                strVar += "                                    <p>Email:<\/p>";
                strVar += "                                    <p>&nbsp;<\/p>";
                strVar += "                                    <p>Phone Number:<\/p>";
                strVar += "                                <\/div>";
                strVar += "                            <\/div>";
                strVar += "                        <\/div>";
                strVar += "                    <\/div>";

                $("#existModalBestScore").text("Giống nhau: " + parseFloat(parseFloat(data["face"]["score"])*100).toFixed(2) + "%");
                $("#myModal .modal-body").append(strVar);
                $("#blacklistKeepTracking").attr("onclick", "interval_check_realtime('"+ cmt_id +"',1,0);");

                $("#myModal").modal("show");
            }
        }
    });
}

function check_fake_user_callback(){
    //người fake
    formData = new FormData();
    cmt_id = $("#chungminhthu").val();
    formData.append("id", cmt_id);

    $.ajax({
        type: "POST",
        url: API + "check_fake",
        data: formData,
        processData: false,
        contentType: false,
        success: function (data) {
            console.log(data);
           if(data["status"] == 0){
                // người mới hoàn toàn
                document.getElementById("u4488_img").src = "static/images/check-mark--crop-u5972.png?crc=3822812632";
                $("#u4993-6 p").text("CLEAN: Người này là người mới");

                $("#label_result").attr("data-target", "#newUserModal");
                $("#label_result").show(); 
            }else if(data["status"] == 1){
                //người fake
                $("#myModal .modal-body").children().remove();

                document.getElementById("u4488_img").src = "static/images/alert-icon-1562.png?crc=523315291";
                $("#u4993-6 p").text("Người này giống "+ parseFloat(parseFloat(data["result"][0]["score"])*100).toFixed(2) +"% với người từng giao dịch trước đây");
                
                $("#label_result").attr("data-target", "#myModal");
                $("#label_result").show(); 

                $("#u3856-4 p").text('NGƯỜI NÀY TỪNG GIAO DỊCH VỚI CMT KHÁC');
                var strVar="";

                for (var i = 0; i < data["result"].length; i++) {
                    strVar += "<div class=\"position_content\" id=\"u3853_position_content\">";
                    strVar += "                        <div class=\"clearfix colelem\" id=\"ppu3858-4\"><!-- group -->";
                    strVar += "                            <div class=\"clearfix grpelem\" id=\"pu3858-4\"><!-- column -->";
                    strVar += "                                <div class=\"clearfix colelem\" id=\"u3858-4\"><!-- content -->";
                    strVar += "                                    <p>NHẬN DẠNG KHUÔN MẶT<\/p>";
                    strVar += "                                <\/div>";
                    strVar += "                                <div class=\"clearfix colelem\" id=\"pu3879\"><!-- group -->";
                    strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3879\"><!-- group -->";
                    strVar += "                                        <div class=\"clearfix grpelem\" id=\"u3892-4\"><!-- content -->";
                    strVar += "                                            <img class=\"img-rounded\" width=\"250\" height=\"160\" style=\"position: absolute;\" src='"+ imagesScanned[0]["src"] +"'>";
                    strVar += "                                        <\/div>";
                    strVar += "                                    <\/div>";
                    strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3883\"><!-- group -->";
                    strVar += "                                        <div class=\"clearfix grpelem\" id=\"u3905-4\"><!-- content -->";
                    strVar += "                                            <img class=\"img-rounded\" width=\"250\" height=\"160\" style=\"position: absolute;\" src='"+ data["result"][i]["image_cmt"].replace("/data/image-processing/facenet/data", "/facenet") +"'>";
                    strVar += "                                        <\/div>";
                    strVar += "                                    <\/div>";
                    strVar += "                                <\/div>";
                    strVar += "                                <div class=\"clearfix colelem\" id=\"pu3881\"><!-- group -->";
                    strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3881\"><!-- group -->";
                    strVar += "                                        <div class=\"clearfix grpelem\" id=\"u3893-4\"><!-- content -->";
                    strVar += "                                            <img class=\"img-rounded\" width=\"250\" height=\"160\" style=\"position: absolute;\" src='"+ data["result"][i]["image_raw"].replace("/data/image-processing/facenet/data", "/facenet") +"'>";
                    strVar += "                                        <\/div>";
                    strVar += "                                    <\/div>";
                    strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3884\"><!-- group -->";
                    strVar += "                                        <div class=\"clearfix grpelem\" id=\"u3895-4\"><!-- content -->";
                    strVar += "                                            <img class=\"img-rounded\" width=\"250\" height=\"160\" style=\"position: absolute;\" src='"+ data["result"][i]["image_query"].replace("/data/image-processing/facenet/data", "/facenet") +"'>";
                    strVar += "                                        <\/div>";
                    strVar += "                                    <\/div>";
                    strVar += "                                <\/div>";
                    strVar += "                                <div class=\"clearfix colelem\" id=\"u3889\"><!-- group -->";
                    strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3897-4\"><!-- content -->";
                    strVar += "                                        <p>Giống nhau "+ parseFloat(parseFloat(data["result"][i]["score"])*100).toFixed(2) +"%<\/p>";
                    strVar += "                                    <\/div>";
                    strVar += "                                <\/div>";
                    strVar += "                            <\/div>";
                    strVar += "                            <div class=\"clearfix grpelem\" id=\"u3859-4\"><!-- content -->";
                    strVar += "                                <p>HỒ SƠ<\/p>";
                    strVar += "                            <\/div>";
                    strVar += "                            <div class=\"grpelem\" id=\"u3899\"><!-- simple frame --><\/div>";
                    strVar += "                            <div class=\"grpelem\" id=\"u3900\"><!-- simple frame --><\/div>";
                    strVar += "                            <div class=\"rgba-background clearfix grpelem\" id=\"u3901\"><!-- column -->";
                    strVar += "                                <div class=\"clearfix colelem\" id=\"pu3903\"><!-- group -->";
                    strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3903\"><!-- group -->";
                    strVar += "                                        <div class=\"clearfix grpelem\" id=\"u3906-4\"><!-- content -->";
                    strVar += "                                            <img class=\"img-rounded\" width=\"250\" height=\"160\" style=\"position: absolute;\" src='"+ data["result"][i]["image_cmt"].replace("/data/image-processing/facenet/data", "/facenet") +"'>";
                    strVar += "                                        <\/div>";
                    strVar += "                                    <\/div>";
                    strVar += "                                    <div class=\"clearfix grpelem\" id=\"u3904\"><!-- group -->";
                    strVar += "                                        <div class=\"clearfix grpelem\" id=\"u3907-4\"><!-- content -->";
                    strVar += "                                            <img class=\"img-rounded\" width=\"250\" height=\"160\" style=\"position: absolute;\" src='"+ data["result"][i]["image_query"].replace("/data/image-processing/facenet/data", "/facenet") +"'>";
                    strVar += "                                        <\/div>";
                    strVar += "                                    <\/div>";
                    strVar += "                                <\/div>";
                    strVar += "                                <div class=\"clearfix colelem\" id=\"u3902-35\"><!-- content -->";
                    strVar += "                                    <p>NAME: "+ data["result"][i]["id_cmt"] +"<\/p>";
                    strVar += "                                    <p>&nbsp;<\/p>";
                    strVar += "                                    <p>Gender:<\/p>";
                    strVar += "                                    <p>&nbsp;<\/p>";
                    strVar += "                                    <p>Date of birth:<\/p>";
                    strVar += "                                    <p>&nbsp;<\/p>";
                    strVar += "                                    <p>Home Adress:<\/p>";
                    strVar += "                                    <p>&nbsp;<\/p>";
                    strVar += "                                    <p>Email:<\/p>";
                    strVar += "                                    <p>&nbsp;<\/p>";
                    strVar += "                                    <p>Phone Number:<\/p>";
                    strVar += "                                <\/div>";
                    strVar += "                            <\/div>";
                    strVar += "                        <\/div>";
                    strVar += "                    <\/div>";

                    break;
                }

                $("#blacklistKeepTracking").attr("onclick", "interval_check_realtime('"+ data["result"][0]["id_cmt"] +"',1,0);");
                $("#myModal .modal-body").append(strVar);
                $("#myModal").modal("show");
            }
        }
    });
}

function interval_check_realtime(id_raw, type, status){
    $(".progress_active").removeClass('progress_active');
    $("#u1506").addClass('progress_active');

    if(status == 0){
        a = "check_realtime_image_vs_image_callback('"+ id_raw +"', '"+ type +"')";
        check_realtime_image_vs_image_callback(id_raw, type);
        setInterval(a, 5000);
    }else{
        a = "check_realtime_image_vs_cmt_callback('"+ id_raw +"', '"+ type +"')";
        check_realtime_image_vs_cmt_callback(id_raw, type);
        setInterval(a, 5000);
    }
    
}

function finish_callback(){
    formData = new FormData();
    cmt_id = $("#chungminhthu").val();
    formData.append("id", cmt_id);

    $.ajax({
        type: "POST",
        url: API + "addToDB",
        data: formData,
        processData: false,
        contentType: false,
        success: function (data) {
            if(data["status"] == 1){
                alert("Tiến hành giao dịch");
                location.reload();
            }
        }
    });
}

function check_realtime_image_vs_image_callback(id_raw, type){
    formData = new FormData();
    cmt_id = $("#chungminhthu").val();
    formData.append("id_query", cmt_id);
    formData.append("id_raw", id_raw);
    formData.append("type", type);

    $.ajax({
        type: "POST",
        url: API + "realtime_image_vs_image",
        data: formData,
        processData: false,
        contentType: false,
        success: function (data) {
            $("#div_realtime").show();
            document.getElementById("img_realtime").src = data["new_image"].replace("/data/image-processing/facenet/data", "/facenet");
            adddata(parseFloat(parseFloat(data["score_cmt"])*100).toFixed(2));
            $("#myChartDiv").show();
        }
    });
}

function check_realtime_image_vs_cmt_callback(id_raw, type){
    formData = new FormData();
    cmt_id = $("#chungminhthu").val();
    formData.append("id_query", cmt_id);
    formData.append("id_raw", id_raw);

    $.ajax({
        type: "POST",
        url: API + "realtime_image_vs_cmt",
        data: formData,
        processData: false,
        contentType: false,
        success: function (data) {
            $("#div_realtime").show();
            document.getElementById("img_realtime").src = data["new_image"].replace("/data/image-processing/facenet/data", "/facenet");
            adddata(parseFloat(parseFloat(data["score_cmt"])*100).toFixed(2));
            $("#myChartDiv").show();
        }
    });
}

//return a promise that resolves with a File instance
function urltoFile(url, filename, mimeType){
    return (fetch(url)
        .then(function(res){return res.arrayBuffer();})
        .then(function(buf){return new File([buf], filename, {type:mimeType});})
    );
}

function addBlackList(){
    formData = new FormData();
    blackname = $("#blackname").val();
    blackcmt = $("#blackcmt").val();
    blackfile = $("#blackfile_cmt")[0].files[0];
    blackfile_face = $("#blackfile_face")[0].files[0];

    formData.append("blackname", blackname);
    formData.append("blackcmt", blackcmt);
    formData.append("myfile", blackfile);
    formData.append("myfile_face", blackfile_face);
    $.ajax({
        type: "POST",
        url: API + "add-black-list",
        data: formData,
        processData: false,
        contentType: false,
        success: function (data) {
            if(data == "success"){
                location.reload();
                
                alert("Thêm danh sách đen thành công");
            }
        }
    });
}
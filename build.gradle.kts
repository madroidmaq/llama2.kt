plugins {
    kotlin("jvm") version "1.9.0"
    application
}

group = "com.madroid.aigc"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}

kotlin {
    jvmToolchain(8)
}

application {
    mainClass.set("Llama2")
}

tasks.register<Copy>("createJarIfNeeded") {
    from(sourceSets.main.get().output)
    into("$buildDir/libs")
    include("Llama2.jar")
    duplicatesStrategy = DuplicatesStrategy.INCLUDE
    doLast {
        if (didWork) {
            println("JAR file created successfully.")
        } else {
            println("JAR file already exists.")
        }
    }
}

tasks.register<JavaExec>("executeMain") {
    dependsOn("createJarIfNeeded")
    mainClass = "Llama2"
    val checkPoint = project.findProperty("cp")?.toString()
    args = listOf(checkPoint)
    classpath = sourceSets.main.get().runtimeClasspath
    doFirst {
        println("Executing main function...")
    }
}

tasks.register("completion") {
    dependsOn("executeMain")
}
